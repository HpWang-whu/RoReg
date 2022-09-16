import os
import time
import torch
import open3d
import numpy as np
from tqdm import tqdm
from utils.r_eval import compute_R_diff, matrix_from_quaternion
from utils.utils import transform_points,SVDR,make_non_exists_dir,to_cuda
from functools import partial
from network import name2network
import multiprocessing
from multiprocessing import Pool

def R_pre_log(dataset,save_dir):
    writer=open(f'{save_dir}/pre.log','w')
    pair_num=int(len(dataset.pc_ids))
    for pair in dataset.pair_ids:
        pc0,pc1=pair
        ransac_result=np.load(f'{save_dir}/{pc0}-{pc1}.npz',allow_pickle=True)
        transform_pr=ransac_result['trans']
        writer.write(f'{int(pc0)}\t{int(pc1)}\t{pair_num}\n')
        writer.write(f'{transform_pr[0][0]}\t{transform_pr[0][1]}\t{transform_pr[0][2]}\t{transform_pr[0][3]}\n')
        writer.write(f'{transform_pr[1][0]}\t{transform_pr[1][1]}\t{transform_pr[1][2]}\t{transform_pr[1][3]}\n')
        writer.write(f'{transform_pr[2][0]}\t{transform_pr[2][1]}\t{transform_pr[2][2]}\t{transform_pr[2][3]}\n')
        writer.write(f'{0.0}\t{0.0}\t{0.0}\t{1.0}\n')
    writer.close()

class refiner:
    def __init__(self):
        pass

    def center_cal(self,key_m0,key_m1,scores):
        key_m0=key_m0*scores[:,None]
        key_m1=key_m1*scores[:,None]
        key_m0=np.sum(key_m0,axis=0)
        key_m1=np.sum(key_m1,axis=0)
        return key_m0,key_m1

    def SVDR_w(self,beforerot,afterrot,scores):# beforerot afterrot Scene2,Scene1
        weight=np.diag(scores)
        H=np.matmul(np.matmul(np.transpose(afterrot),weight),beforerot)
        U,Sigma,VT=np.linalg.svd(H)
        return np.matmul(U,VT)

    def R_cal(self,key_m0,key_m1,center0,center1,scores):
        key_m0=key_m0-center0[None,:]
        key_m1=key_m1-center1[None,:]
        return self.SVDR_w(key_m1,key_m0,scores)

    def t_cal(self,center0,center1,R):
        return center0-center1@R.T

    def Rt_cal(self,key_m0,key_m1,scores):
        scores=scores/np.sum(scores)
        center0,center1=self.center_cal(key_m0,key_m1,scores)
        R=self.R_cal(key_m0,key_m1,center0,center1,scores)
        t=self.t_cal(center0,center1,R)
        return R,t
    
    def Refine_trans(self,key_m0,key_m1,T,scores,inlinerdist=None):
        key_m1_t=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1_t),axis=-1)
        overlap=np.where(diff<inlinerdist*inlinerdist)[0]
            
        scores=scores[overlap]
        key_m0=key_m0[overlap]
        key_m1=key_m1[overlap]
        R,t=self.Rt_cal(key_m0, key_m1, scores)
        Tnew=np.eye(4)
        Tnew[0:3,0:3]=R
        Tnew[0:3,3]=t
        return Tnew

#yohoc
class extractor_dr_index:
    def __init__(self,cfg):
        self.cfg=cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').reshape([-1]).astype(np.int)).cuda()

    def Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        des1_eqv=des1_eqv[:,self.Nei_in_SO3].reshape([-1,60,60])
        cor=torch.einsum('fag,fg->a',des1_eqv,des2_eqv)
        return torch.argmax(cor)

    def Batch_Des2R_torch(self,des1_eqv,des2_eqv):#beforerot afterrot
        B,F,G=des1_eqv.shape
        des1_eqv=des1_eqv[:,:,self.Nei_in_SO3].reshape([B,F,60,60])
        cor=torch.einsum('bfag,bfg->ba',des1_eqv,des2_eqv)
        return torch.argmax(cor,dim=1)
  
    def Rindex(self,dataset,keynum):
        match_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        Save_dir=f'{match_dir}/DR_index'
        make_non_exists_dir(Save_dir)
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature'
        
        print(f'extract the drindex of the matches on {dataset.name}')
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            # if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            match_pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            feats0=torch.from_numpy(feats0[match_pps[:,0]].astype(np.float32)).cuda()
            feats1=torch.from_numpy(feats1[match_pps[:,1]].astype(np.float32)).cuda()
            pre_idxs=self.Batch_Des2R_torch(feats1,feats0).cpu().numpy()
            np.save(f'{Save_dir}/{id0}-{id1}.npy',pre_idxs)

class yohoc_ransac:
    def __init__(self,cfg):
        self.cfg=cfg
        self.inliner_dist=cfg.ransac_ird
        self.refiner = refiner()
    
    def DR_statictic(self,DR_indexs):
        R_index_pre_statistic={}
        for i in range(60):
            R_index_pre_statistic[i]=[]
        for t in range(DR_indexs.shape[0]):
            R_index_pre_statistic[DR_indexs[t]].append(t)
        R_index_pre_probability=[]
        for i in range(60):
            if len(R_index_pre_statistic[i])<2:
                R_index_pre_probability.append(0)
            else:
                num=float(len(R_index_pre_statistic[i]))/100.0
                R_index_pre_probability.append(num*(num-0.01)*(num-0.02))
        R_index_pre_probability=np.array(R_index_pre_probability)
        if np.sum(R_index_pre_probability)==0:
            return None,np.zeros(60)
        else:
            R_index_pre_probability=R_index_pre_probability/np.sum(R_index_pre_probability)
            return R_index_pre_statistic,R_index_pre_probability

    def Threepps2Tran(self,kps0_init,kps1_init):
        center0=np.mean(kps0_init,0,keepdims=True)
        center1=np.mean(kps1_init,0,keepdims=True)
        m = (kps1_init-center1).T @ (kps0_init-center0)
        U,S,VT = np.linalg.svd(m)
        rotation = VT.T @ U.T   #predicted RT
        offset = center0 - (center1 @ rotation.T)
        transform=np.concatenate([rotation,offset.T],1)
        return transform #3*4


    def overlap_cal(self,key_m0,key_m1,T,scores):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        overlap=np.where(diff<self.inliner_dist*self.inliner_dist)[0]
        overlap=np.sum(scores[overlap])/scores.shape[0]
        return overlap

    def transdiff(self,gt,pre):
        Rdiff=compute_R_diff(gt[0:3:,0:3],pre[0:3:,0:3])
        tdiff=np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff


    def ransac_once(self,dataset,keynum,max_iter,pair):
        match_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        Index_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/yohoc/{max_iter}iters'

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name

        id0,id1=pair
        #if os.path.exists(f'{Save_dir}/{id0}-{id1}.npz'):continue
        gt=dataset.get_transform(id0,id1)
        #Keypoints
        Keys0=dataset.get_kps(id0)
        Keys1=dataset.get_kps(id1)
        
        #scores
        scores=np.load(f'{match_dir}/scores/{id0}-{id1}.npy')
        pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
        Keys_m0_init=Keys0[pps[:,0]]
        Keys_m1_init=Keys1[pps[:,1]]
        
        #Key_pps

        sample_index = np.arange(pps.shape[0])
        if self.cfg.RM:
            if self.cfg.match_n <0.999:
                num=max(scores.shape[0]*self.cfg.match_n,10)
            else:
                num = self.cfg.match_n
            sample_index=np.argsort(scores)[-int(num):]
            
        pps = pps[sample_index]
        Keys_m0=Keys_m0_init[sample_index]
        Keys_m1=Keys_m1_init[sample_index]

        #Indexs
        Index=np.load(f'{Index_dir}/{id0}-{id1}.npy')[sample_index]
        #DR_statistic
        R_index_pre_statistic,R_index_pre_probability=self.DR_statictic(Index)


        #RANSAC
        iter_ransac=0
        recall_time=0
        best_overlap=0
        best_trans_ransac=np.ones(4)
        best_3p_in_0=np.ones([3,3])
        best_3p_in_1=np.ones([3,3])
        max_time=50000
        exec_time=0

        if np.sum(R_index_pre_probability)<1e-5:
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=np.random.rand(4,4), center=np.concatenate([best_3p_in_0,best_3p_in_1],axis=0),recalltime=50000)
            return 0


        while iter_ransac<max_iter:
            if exec_time>max_time:break
            exec_time+=1
            R_index=np.random.choice(range(60),p=R_index_pre_probability)
            if (len(R_index_pre_statistic[R_index])<2):
                continue
            iter_ransac+=1
            idxs_init=np.random.choice(np.array(R_index_pre_statistic[R_index]),3) #guarantee the same index
            kps0_init=Keys_m0[idxs_init]
            kps1_init=Keys_m1[idxs_init]
            trans=self.Threepps2Tran(kps0_init,kps1_init)
            overlap=self.overlap_cal(Keys_m0_init,Keys_m1_init,trans,scores)
            if overlap>best_overlap:
                best_overlap=overlap
                best_trans_ransac=trans
                best_3p_in_0=kps0_init
                best_3p_in_1=kps1_init
                recall_time=iter_ransac
        #save:
        best_trans_ransac=self.refiner.Refine_trans(Keys_m0_init,Keys_m1_init,best_trans_ransac,scores,inlinerdist=self.inliner_dist*2.0)
        best_trans_ransac=self.refiner.Refine_trans(Keys_m0_init,Keys_m1_init,best_trans_ransac,scores,inlinerdist=self.inliner_dist)
        np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=best_trans_ransac,recalltime=recall_time)


    def ransac(self,dataset,keynum,max_iter=1000):
        match_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        Index_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/yohoc/{max_iter}iters'
        make_non_exists_dir(Save_dir)

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name

        print(f'Ransac with YOHO-C on {dataset.name}:')
        pair_ids=dataset.pair_ids
        pool = Pool(len(pair_ids))
        func = partial(self.ransac_once,dataset,keynum,max_iter)
        pool.map(func,pair_ids)
        pool.close()
        pool.join()
        R_pre_log(dataset,Save_dir)
        print('Done')

class yohoc:
    def __init__(self, cfg):
        self.rind_extractor = extractor_dr_index(cfg)
        self.ransacer = yohoc_ransac(cfg)
    def run(self,dataset,keynum,max_iter):
        self.rind_extractor.Rindex(dataset, keynum)
        self.ransacer.ransac(dataset, keynum, max_iter)
    
#yohoo    
class extractor_localtrans():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network['ET_test'](self.cfg).cuda()
        self.best_model_fn=f'{self.cfg.model_fn}/ET/model_best.pth'
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.test_batch_size = self.cfg.bs_ET

    #Model_import
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'],strict=False)
        else:
            raise ValueError("No model exists")
    
    def batch_create(self,feats0_fcgf,feats1_fcgf,feats0_yomo,feats1_yomo,index_pre,start,end):
        #attention: here feats0->feats1_in_batch for it is afterrot
        feats0_fcgf=torch.from_numpy(feats0_fcgf[start:end,:,:].astype(np.float32))
        feats1_fcgf=torch.from_numpy(feats1_fcgf[start:end,:,:].astype(np.float32))
        feats0_yomo=torch.from_numpy(feats0_yomo[start:end,:,:].astype(np.float32))
        feats1_yomo=torch.from_numpy(feats1_yomo[start:end,:,:].astype(np.float32))
        index_pre=torch.from_numpy(index_pre[start:end].astype(np.int))
        return {
                'before_eqv0':feats1_fcgf,#exchanged
                'before_eqv1':feats0_fcgf,
                'after_eqv0':feats1_yomo,
                'after_eqv1':feats0_yomo,
                'pre_idx':index_pre
        }

    def Rt_pre(self,dataset,keynum):
        self._load_model()
        self.network.eval()
        #dataset: (5000*32*60->pp*32*60)*4 + pre_index_trans-> pp*128*60
        match_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        DRindex_dir=f'{match_dir}/DR_index'
        Save_dir=f'{match_dir}/Trans_pre'
        make_non_exists_dir(Save_dir)
        
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        FCGF_dir=f'{self.cfg.output_cache_fn}/{datasetname}/{self.cfg.backbone}_Input_Group_feature'
        YOMO_dir=f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature'

        #feat1:beforrot feat0:afterrot
        print(f'Extracting the local transformation on each correspondence of {dataset.name}')       
        alltime=0 
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            #if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            
            feats0_fcgf=np.load(f'{FCGF_dir}/{id0}.npy')[pps[:,0],:,:] #pps*32*60
            feats1_fcgf=np.load(f'{FCGF_dir}/{id1}.npy')[pps[:,1],:,:] #pps*32*60
            feats0_yomo=np.load(f'{YOMO_dir}/{id0}.npy')[pps[:,0],:,:] #pps*32*60
            feats1_yomo=np.load(f'{YOMO_dir}/{id1}.npy')[pps[:,1],:,:] #pps*32*60
            Index_pre=np.load(f'{DRindex_dir}/{id0}-{id1}.npy')        #pps

            Keys0=dataset.get_kps(id0)[pps[:,0],:]  #pps*3
            Keys1=dataset.get_kps(id1)[pps[:,1],:]  #pps*3

            bi=0
            Rs=[]
            while(bi*self.test_batch_size<feats0_fcgf.shape[0]):
                start=bi*self.test_batch_size
                end=(bi+1)*self.test_batch_size
                batch=self.batch_create(feats0_fcgf,feats1_fcgf,feats0_yomo,feats1_yomo,Index_pre,start,end)
                batch=to_cuda(batch)
                with torch.no_grad():
                    batch_output=self.network(batch)
                bi+=1
                deltaR=batch_output['quaternion_pre'].cpu().numpy()
                anchorR=batch_output['pre_idxs'].cpu().numpy()
                for i in range(deltaR.shape[0]):
                    R_residual=matrix_from_quaternion(deltaR[i])
                    R_anchor=self.Rgroup[int(anchorR[i])]
                    Rs.append((R_residual@R_anchor)[None,:,:])
            Rs=np.concatenate(Rs,axis=0) #pps*3*3
            Trans=[]
            for R_id in range(Rs.shape[0]):
                R=Rs[R_id]
                key0=Keys0[R_id] #after rot key0=t+key1@R.T
                key1=Keys1[R_id] #before rot
                t=key0-key1@R.T
                trans_one=np.concatenate([R,t[:,None]],axis=1)
                Trans.append(trans_one[None,:,:])
            Trans=np.concatenate(Trans,axis=0)
            np.save(f'{Save_dir}/{id0}-{id1}.npy',Trans)
               
class yohoo_ransac:
    def __init__(self,cfg):
        self.cfg=cfg
        self.inliner_dist=cfg.ransac_ird
        self.Nei_in_SO3=np.load(f'{self.cfg.SO3_related_files}/60_60.npy')
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy')
        self.refiner = refiner()

    def overlap_cal(self,key_m0,key_m1,T,scores):
        key_m1=transform_points(key_m1,T)
        diff=np.sum(np.square(key_m0-key_m1),axis=-1)
        overlap=np.where(diff<self.inliner_dist*self.inliner_dist)[0]
        overlap=np.sum(scores[overlap])/scores.shape[0]
        return overlap

    def transdiff(self,gt,pre):
        Rdiff=compute_R_diff(gt[0:3:,0:3],pre[0:3:,0:3])
        tdiff=np.sqrt(np.sum(np.square(gt[0:3,3]-pre[0:3,3])))
        return Rdiff,tdiff


    def ransac(self,dataset,keynum,max_iter=1000):
        match_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        Trans_dir=f'{match_dir}/Trans_pre'
        Save_dir=f'{match_dir}/yohoo/{max_iter}iters'
        make_non_exists_dir(Save_dir)

        print(f'Ransac with YOHO-O on {dataset.name}:')
        for pair in tqdm(dataset.pair_ids):
            id0,id1=pair
            gt=dataset.get_transform(id0,id1)
            # if os.path.exists(f'{Save_dir}/{id0}-{id1}.npz'):continue
            #Keypoints
            Keys0=dataset.get_kps(id0)
            Keys1=dataset.get_kps(id1)
            
            #scores
            scores=np.load(f'{match_dir}/scores/{id0}-{id1}.npy')
            
            #Key_pps
            pps=np.load(f'{match_dir}/{id0}-{id1}.npy')
            Keys_m0=Keys0[pps[:,0]]
            Keys_m1=Keys1[pps[:,1]]
            #Indexs
            Trans=np.load(f'{Trans_dir}/{id0}-{id1}.npy')

            if self.cfg.RM:
                if self.cfg.match_n <0.999:
                    num=max(scores.shape[0]*self.cfg.match_n,10)
                else:
                    num = self.cfg.match_n
                sample_index=np.argsort(scores)[-int(num):]
                Trans=Trans[sample_index]

            index=np.arange(Trans.shape[0])
            np.random.shuffle(index)
            Trans_ransac=Trans[index[0:max_iter]] #if max_iter>index.shape[0] then =index.shape[0] automatically
            #RANSAC
            recall_time=0
            best_overlap=0
            best_trans_ransac=0
            for t_id in range(Trans_ransac.shape[0]):
                T=Trans_ransac[t_id]
                overlap=self.overlap_cal(Keys_m0,Keys_m1,T,scores)
                if overlap>best_overlap:
                    best_overlap=overlap
                    best_trans_ransac=T
                    recall_time=t_id
            # refine:
            best_trans_ransac=self.refiner.Refine_trans(Keys_m0,Keys_m1,best_trans_ransac,scores,inlinerdist=self.inliner_dist*2.0)
            best_trans_ransac=self.refiner.Refine_trans(Keys_m0,Keys_m1,best_trans_ransac,scores,inlinerdist=self.inliner_dist)
            #save
            np.savez(f'{Save_dir}/{id0}-{id1}.npz',trans=best_trans_ransac, recalltime=recall_time)

        R_pre_log(dataset,Save_dir)

class yohoo:
    def __init__(self, cfg):
        self.cfg = cfg
        self.rind_extractor = extractor_dr_index(cfg)
        self.localT_extractor = extractor_localtrans(cfg)
        self.ransacer = yohoo_ransac(cfg)
    def run(self,dataset,keynum,max_iter):
        self.rind_extractor.Rindex(dataset, keynum)
        self.localT_extractor.Rt_pre(dataset, keynum)
        self.ransacer.ransac(dataset, keynum, max_iter)
    

import os,sys
sys.path.append('..')
import numpy as np
import torch
import tqdm
import time
from utils.utils import make_non_exists_dir,to_cuda
from utils.knn_search import knn_module
from network import name2network

class NMS_sample():
    def __init__(self, num, k):
        '''Non-maximum suppression'''
        self.k=k #neighborhood
        self.num=num
        self.KNN=knn_module.KNN(self.k)

    def sample(self,keys,scores):
        if keys.shape[0]<self.num:
            return np.arange(keys.shape[0])
        keys=torch.from_numpy(keys.astype(np.float32)[None,:,:]).permute(0,2,1).cuda()
        d,argmin=self.KNN(keys,keys)
        argmin=argmin[0].permute(1,0).cpu().numpy()#5000*5
        scores_nei=scores[argmin.reshape(-1)].reshape(-1,self.k)
        nei_max=np.max(scores_nei,axis=-1)
        sam_indexs=np.where(scores>=nei_max)[0]

        if sam_indexs.shape[0]>self.num:
            sam_scores=scores[sam_indexs]
            sam_scores=sam_scores/np.sum(sam_scores)
            resam_indexs=np.argsort(sam_scores)[-self.num:]
            sam_indexs=sam_indexs[resam_indexs]

        if sam_indexs.shape[0]<self.num:
            left=self.num-sam_indexs.shape[0]
            index_left=np.where(scores<nei_max)[0]
            scores_left=scores[index_left]
            left_index=np.argsort(scores_left)[-left:]
            left_index=index_left[left_index]
            sam_indexs=np.concatenate([sam_indexs,left_index],axis=0)

        return sam_indexs

class mutual():
    def __init__(self,cfg):
        self.cfg=cfg
        self.KNN=knn_module.KNN(1)

    def run(self,dataset,keynum=5000):
        self.sampler=NMS_sample(keynum,5)
        print(f'Matching the keypoints with mutual on {dataset.name}')
        Save_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        make_non_exists_dir(Save_dir)
        Save_score_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}/scores'
        make_non_exists_dir(Save_score_dir)

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature'
        alltime=0
        for pair in tqdm.tqdm(dataset.pair_ids):
            id0,id1=pair
            # if os.path.exists(f'{Save_dir}/{id0}-{id1}.npy'):continue
            
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            feats0=np.mean(feats0,axis=-1).astype(np.float32)
            feats1=np.mean(feats1,axis=-1).astype(np.float32)
            feats0 = feats0/(np.sqrt(np.sum(np.square(feats0),axis=1,keepdims=True))+1e-5)
            feats1 = feats1/(np.sqrt(np.sum(np.square(feats1),axis=1,keepdims=True))+1e-5)
            keys0=dataset.get_kps(id0)
            keys1=dataset.get_kps(id1)

            if self.cfg.RD:
                det_scores0=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/det_score/{id0}.npy')
                det_scores1=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/det_score/{id1}.npy')
                sample0=self.sampler.sample(keys0,det_scores0)
                sample1=self.sampler.sample(keys1,det_scores1)

            else:
                sample0=np.arange(feats0.shape[0])
                sample1=np.arange(feats1.shape[0])
                np.random.shuffle(sample0)
                np.random.shuffle(sample1)
                sample0=sample0[0:keynum]
                sample1=sample1[0:keynum]

            feats0=feats0[sample0]
            feats1=feats1[sample1]
            feats0=torch.from_numpy(np.transpose(feats0)[None,:,:]).cuda()
            feats1=torch.from_numpy(np.transpose(feats1)[None,:,:]).cuda()
            d,argmin_of_0_in_1=self.KNN(feats1,feats0)
            argmin_of_0_in_1=argmin_of_0_in_1[0,0].cpu().numpy()
            d,argmin_of_1_in_0=self.KNN(feats0,feats1)
            argmin_of_1_in_0=argmin_of_1_in_0[0,0].cpu().numpy()
            match_pps=[]
            for i in range(argmin_of_0_in_1.shape[0]):
                in0=i
                in1=argmin_of_0_in_1[i]
                inv_in0=argmin_of_1_in_0[in1]
                if in0==inv_in0:
                    match_pps.append(np.array([[in0,in1]]))
            match_pps=np.concatenate(match_pps,axis=0)
            match_pps[:,0]=sample0[match_pps[:,0]]
            match_pps[:,1]=sample1[match_pps[:,1]]
            np.save(f'{Save_dir}/{id0}-{id1}.npy',match_pps)
            np.save(f'{Save_score_dir}/{id0}-{id1}.npy',np.ones(match_pps.shape[0]))

class yoho_mat():
    def __init__(self,cfg):
        self.cfg=cfg
        coor_norm_step=0.025
        self.network=name2network['RM_test'](self.cfg).cuda()
        self.best_model_fn=f'{self.cfg.model_fn}/RM/model_best.pth'
        self.KNN=knn_module.KNN(1)
        self._load_model()
    
    #Model_import
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'],strict=True)
            # print(f'Resuming best para {best_para}')
        else:
            raise ValueError("No model exists")

    def get_ot_match(self,batch):
        pairs=[]
        batch=to_cuda(batch)
        self.network.eval()
        with torch.no_grad():
            result=self.network(batch)
        matches0=result['matches0'][0].cpu().numpy()
        scores=result['matching_scores0'][0].cpu().numpy()
        scores0=scores
        scores1=result['matching_scores1'][0].cpu().numpy()
        score_ms=[]
        for i in range(matches0.shape[0]):
            if not matches0[i]==-1:
                pairs.append(np.array([[i,matches0[i]]]))
                score_ms.append(scores[i])
        if len(pairs)<3:
            pairs=None
        else:
            pairs=np.concatenate(pairs,axis=0)
        return pairs,np.array(score_ms),scores0,scores1
    
    def run(self,dataset,keynum=2500):
        self.sampler=NMS_sample(keynum,5)
        Save_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}'
        make_non_exists_dir(Save_dir)
        Save_score_dir=f'{self.cfg.output_cache_fn}/{dataset.name}/match_{keynum}/scores'
        make_non_exists_dir(Save_score_dir)

        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        Feature_dir=f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature'
        alltime=0
        print(f'Matching the keypoints with rotation coherence matcher on {dataset.name}')
        for pair in tqdm.tqdm(dataset.pair_ids):
            id0,id1=pair
            # if os.path.exists(f'{Save_score_dir}/{id0}-{id1}.npy'):continue
            feats0=np.load(f'{Feature_dir}/{id0}.npy') #5000,32,60
            feats1=np.load(f'{Feature_dir}/{id1}.npy') #5000,32,60
            keys0=dataset.get_kps(id0)
            keys1=dataset.get_kps(id1)
            
            if self.cfg.RD:
                det_scores0=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/det_score/{id0}.npy')
                det_scores1=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/det_score/{id1}.npy')
                sample0=self.sampler.sample(keys0,det_scores0)
                sample1=self.sampler.sample(keys1,det_scores1)
            else:
                sample0=np.arange(feats0.shape[0])
                sample1=np.arange(feats1.shape[0])
                np.random.shuffle(sample0)
                np.random.shuffle(sample1)
                sample0=sample0[0:keynum]
                sample1=sample1[0:keynum]

            feats0=feats0[sample0]
            feats1=feats1[sample1]
            keys0=dataset.get_kps(id0)[sample0]
            keys1=dataset.get_kps(id1)[sample1]
                
            batch={
                'feats0':torch.from_numpy(feats1[None,:,:,:].astype(np.float32)),
                'feats1':torch.from_numpy(feats0[None,:,:,:].astype(np.float32)),
                'keys0':torch.from_numpy(keys1[None,:,:].astype(np.float32)),
                'keys1':torch.from_numpy(keys0[None,:,:].astype(np.float32))
            }

            matches,scores,scores1,scores0=self.get_ot_match(batch)
            if matches is None:
                matches=np.ones(1,2)
                scores=np.ones(1)

            matches_in_former=[]
            matches_in_former.append(sample0[matches[:,1]][:,None])#note: 1 here is the pc0
            matches_in_former.append(sample1[matches[:,0]][:,None])
            matches_in_former=np.concatenate(matches_in_former,axis=1)

            np.save(f'{Save_dir}/{id0}-{id1}.npy',matches_in_former)
            np.save(f'{Save_score_dir}/{id0}-{id1}.npy',scores)

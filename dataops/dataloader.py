import numpy as np
import time
import torch
from torch.utils.data import Dataset
from dataops.dataset import get_dataset_name
from utils.r_eval import *
from utils.utils import read_pickle, transform_points

#train dataset of group feature extractor
class trainset_GF(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.trainlist) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.vallist)[0:3000]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    def __getitem__(self,index):
        if self.is_training:
            item=torch.load(f'{self.output_dir}/trainset/{index}.pth')
            return item
        
        else:
            item=torch.load(f'{self.output_dir}/valset/{index}.pth')
            return item
        

    def __len__(self):
        return len(self.name_pair_ids)

#train dataset of rotatioin guided detector
class trainset_RD(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.batchsize=128
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.datasets=get_dataset_name(self.cfg.trainset, self.cfg.origin_data_dir)
        self.setname=self.cfg.trainset
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.trainlist) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.vallist)[0:500]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    
    def __getitem__(self,index):
        np.random.seed(int(time.time()))
        scenename,pc0,pc1=self.name_pair_ids[index]
        gt=self.datasets[scenename].get_transform(pc0,pc1)
        feats0=np.load(f'{self.output_dir}/{self.setname}/{scenename}/YOHO_Output_Group_feature/{pc0}.npy')
        feats1=np.load(f'{self.output_dir}/{self.setname}/{scenename}/YOHO_Output_Group_feature/{pc1}.npy')
        matches=np.load(f'{self.output_dir}/{self.setname}/{scenename}/match_5000/{pc0}-{pc1}.npy')
        Trans=np.load(f'{self.output_dir}/{self.setname}/{scenename}/match_5000/Trans_pre/{pc0}-{pc1}.npy')
        
        if self.is_training:
            samples=np.arange(Trans.shape[0])
            np.random.shuffle(samples)
            samples=samples[0:self.batchsize]
            matches=matches[samples]
            Trans=Trans[samples]        
        
        feats0=feats0[matches[:,0]]
        feats1=feats1[matches[:,1]]
                
        # RD loss V1 in paper more hyperparameters
        # distinctiveness check
        # keys0=self.datasets[scenename].get_kps(pc0)[matches[:,0]]
        # keys1=self.datasets[scenename].get_kps(pc1)[matches[:,1]]
        # keys1=transform_points(keys1,gt)
        # dist=np.sqrt(np.sum(np.square(keys0-keys1),axis=-1))
        # # rotation check
        # Rdiffs=[]
        # for tid in range(Trans.shape[0]):
        #     T=Trans[tid]
        #     Rdiff=compute_R_diff(T[0:3,0:3],gt[0:3,0:3])
        #     Rdiff=Rdiff-5
        #     if Rdiff<0:
        #         Rdiff=0           
        #     if (Rdiff<25) and (dist[tid]>0.3):
        #         Rdiff=40
        #     Rdiffs.append(Rdiff/60)
        # Rdiffs=torch.from_numpy(np.array(Rdiffs).astype(np.float32))
        
        # RD loss V2 -- fewer hyperparameters + check SE(3) distances only --> more stable convergence
        gt_q = quaternion_from_matrix(gt[0:3,0:3])
        Rdiffs, tdiffs = [], []
        for tid in range(Trans.shape[0]):
            T=Trans[tid]
            # rotation difference in quaternion.
            q = quaternion_from_matrix(T[0:3,0:3])
            Rdiffs.append(np.sqrt(np.sum(np.square(q-gt_q))))
            # translation difference in meters.
            tdiff = np.sum(np.square(T[0:3,3]-gt[0:3,3]))
            tdiffs.append(tdiff)
        Rdiffs = np.array(Rdiffs) + np.array(tdiffs)/3
        Rdiffs=torch.from_numpy(np.array(Rdiffs).astype(np.float32))

        item={
            'feats0':torch.from_numpy(feats0.astype(np.float32)),
            'feats1':torch.from_numpy(feats1.astype(np.float32)),
            'Rdiffs':Rdiffs
        }
        return item

    def __len__(self):
        return len(self.name_pair_ids)

#train dataset of rotatioin coherence matcher
class trainset_RM(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.group=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy')
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.trainlist) #list: name id0 id1 Ri Rj
        else:
            self.name_pair_ids=read_pickle(cfg.vallist)[0:200]   #list: name id0 id1 Ri Rj
    
    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.group.shape[0]):
            R_diff=compute_R_diff(self.group[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id

    def __getitem__(self,index):
        if self.is_training:
            batch_fn=f'{self.output_dir}/RM/Train/{index}.pth'
        else:
            batch_fn=f'{self.output_dir}/RM/Val/{index}.pth'
        item=torch.load(batch_fn)
        if item['pairs'].shape[0]==0:
            return self.__getitem__(index+1)
        R=item['R'].numpy()
        index_T=self.R2DR_id(R.T)
        item['true_idx_T']=torch.from_numpy(np.array([index_T]).astype(np.int))
        return item

    def __len__(self):
        return len(self.name_pair_ids)
    
# train dataset of the equivariance guided transformation estimator
class trainset_ET(Dataset):
    def __init__(self,cfg,is_training=True):
        self.cfg=cfg
        self.output_dir=self.cfg.output_cache_fn
        self.is_training=is_training
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        if self.is_training:
            self.name_pair_ids=read_pickle(cfg.trainlist) #list: name id0 id1 pt1 pt2
        else:
            self.name_pair_ids=read_pickle(cfg.vallist)[0:3000]   #list: name id0 id1 pt1 pt2

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id
    
    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)

    def __getitem__(self,index):
        if self.is_training:
            item=torch.load(f'{self.output_dir}/trainset/{index}.pth')
            return item
        
        else:
            item=torch.load(f'{self.output_dir}/valset/{index}.pth')
            deltaR=self.DeltaR(item['R'].numpy(),int(item['true_idx']))
            item['deltaR']=torch.from_numpy(deltaR.astype(np.float32))
            return item
        

    def __len__(self):
        return len(self.name_pair_ids)
        
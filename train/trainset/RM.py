"""
Trainset preparation for RD and RM.

For data argumentation, we need to add random rotations to scans firstly and keep scan pairs with overlap over 10%.
Already stored in 3dm_train_rot.

Here we need to execute the RD trainset generation firstly.

For RM:
    construct 50k pairs, per pair:

    #sample
    index0 = sample[256,1538] 
    index1 = sample[256,1538] 
    keys0[index0] --> keys0
    keys1[index1] --> keys1
    feats0[index0] --> feats0
    feats1[index1] --> feats1

    #from pc0 to pc1
    R=R1@R0.T

    #pairs
    keys0_gt=keys0@R.T
    keys0_gt,keys1-->dist+matmul+threshold(<0.2)-->pair indexs
    unpair0,unpair1
    matches0(unpair is set as -1)


    #Rargument to pc1 feats1
    Rarguindex-->random[0,59]
    Rargu=Group[Rarguindex]
    Pargu=Permutation[Rarguindex]
    keys1=keys1@Rargu.T
    feats1=feats1[Pargu]

    #delta between pc0 pc1
    R=Rargu@R
    R-->Rindex

    #targument
    t0=np.random.rand(1,3)-0.5
    t1=np.random.rand(1,3)-0.5
    keys0+=t0
    keys1+=t1

    #batch save:
    'keys0': keys0
    'keys1': keys1
    'feats0': feats0
    'feats1': feats1
    'pairs': pairs
    'unpair0': unpair0
    'unpair1': unpair1
    'matches0': matches0 #for validation
    'R': R
    'true_idx': Rindex

"""
import os
import torch
import random
import numpy as np
from tqdm import tqdm
from dataops.dataset import get_dataset_name
from utils.r_eval import compute_R_diff
from parses.parses_rm import get_config
from utils.utils import transform_points,read_pickle,save_pickle,make_non_exists_dir


class RM_trainset():
    def __init__(self,cfg):
        self.cfg=cfg
        self.datasets=get_dataset_name(self.cfg.trainset,self.cfg.origin_data_dir)
        self.group=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy')
        self.perms=np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)
        #RM
        self.min_ps=self.cfg.min_ps
        self.max_ps=self.cfg.max_ps
        self.pps_thre=self.cfg.pps_thre
        #save
        self.trainRM_fn=f'{self.cfg.output_cache_fn}/RM/train_RM.pkl'
        self.valRM_fn=f'{self.cfg.output_cache_fn}/RM/val_RM.pkl'
        self.RMtrain_dir=f'{self.cfg.output_cache_fn}/RM/Train'
        make_non_exists_dir(self.RMtrain_dir)
        self.RMval_dir=f'{self.cfg.output_cache_fn}/RM/Val'
        make_non_exists_dir(self.RMval_dir)
    
    #RM
    def TV_RM(self,batchnum=30000):
        train_pcp=[]
        if not os.path.exists(self.trainRM_fn):
            for name,dataset in tqdm(self.datasets.items()):
                if name in ['wholesetname','valscenes']:continue
                if name in self.datasets['valscenes']:continue
                for pair in dataset.pair_ids:
                    pc0,pc1=pair
                    train_pcp.append((name,pc0,pc1))
            #enough num
            repeat=int(batchnum/len(train_pcp))+1
            train_pcp=[val for val in train_pcp for i in range(repeat)]
            random.shuffle(train_pcp)
            train_pcp=train_pcp[0:batchnum]
            save_pickle(train_pcp,self.trainRM_fn)
        else:
            train_pcp=read_pickle(self.trainRM_fn)

        val_pcp=[]  
        if not os.path.exists(self.valRM_fn):
            for name in self.datasets['valscenes']:
                dataset=self.datasets[name]
                for pair in dataset.pair_ids:
                    pc0,pc1=pair
                    val_pcp.append((name,pc0,pc1))
            random.shuffle(val_pcp)
            save_pickle(val_pcp,self.valRM_fn)
        else:
            val_pcp=read_pickle(self.valRM_fn)

        return train_pcp,val_pcp

    def Keysample(self,keys,feats):
        num=np.random.choice(np.arange(self.min_ps,self.max_ps),1)[0]
        indexs=np.arange(keys.shape[0])
        np.random.shuffle(indexs)
        indexs=indexs[0:num]
        keys=keys[indexs]
        feats=feats[indexs]
        return keys,feats

    def pairmatch(self,keys0,keys1,R):
        keys0_gt=keys0@R.T
        keys0_gt=torch.from_numpy(keys0_gt.astype(np.float32)).cuda()
        keys1=torch.from_numpy(keys1.astype(np.float32)).cuda()
        dist=torch.norm((keys0_gt[:,None,:]-keys1[None,:,:]),dim=-1).cpu().numpy()
        argmin_of_0_in_1=np.argmin(dist,axis=1) #choose_num_0
        argmin_of_1_in_0=np.argmin(dist,axis=0)
        pairs=[]
        unpair0=[]
        unpair1=[]
        matches0=-np.ones(keys0.shape[0])
        matches1=-np.ones(keys1.shape[0])
        for i in range(argmin_of_0_in_1.shape[0]):
            if i==argmin_of_1_in_0[argmin_of_0_in_1[i]]:
                if dist[i,argmin_of_0_in_1[i]]<self.pps_thre:
                    pairs.append([i,argmin_of_0_in_1[i]])
                    matches0[i]=argmin_of_0_in_1[i]
                    matches1[argmin_of_0_in_1[i]]=i
        unpair0=np.where(matches0==-1)[0]
        unpair1=np.where(matches1==-1)[0]
        pairs=np.array(pairs)
        return pairs,unpair0,unpair1,matches0
    
    def Rargu_on1(self,keys1,feats1):
        Rarg_id=np.random.choice(np.arange(60),1)[0]
        Rarg=self.group[Rarg_id]
        Parg=self.perms[Rarg_id]
        keys1=keys1@Rarg.T
        feats1=feats1[:,:,Parg]
        return Rarg,keys1,feats1
    
    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.group.shape[0]):
            R_diff=compute_R_diff(self.group[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id
    
    def batch_generator(self,name,pc0,pc1):
        #readin
        dataset=self.datasets[name]
        keys0=dataset.get_kps(pc0)
        keys1=dataset.get_kps(pc1)
        feats0=np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/YOHO_Output_Group_feature/{pc0}.npy')
        feats1=np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/YOHO_Output_Group_feature/{pc1}.npy')
        Rs=np.load(f'{self.cfg.origin_data_dir}/{dataset.name}/PointCloud/Rs.npy')
        R0=Rs[int(pc0)]
        R1=Rs[int(pc1)]
        #sample
        keys0,feats0=self.Keysample(keys0,feats0)
        keys1,feats1=self.Keysample(keys1,feats1)
        #R,from keys0 to keys1
        R=R1@R0.T
        #pairs,unpair0,unpair1,matches0
        pairs,unpair0,unpair1,matches0=self.pairmatch(keys0,keys1,R)
        #Rargu
        Rarg,keys1,feats1=self.Rargu_on1(keys1,feats1)
        #updateR
        R=Rarg@R
        R_index=self.R2DR_id(R)
        #targu
        t0=np.random.rand(1,3)-0.5
        t1=np.random.rand(1,3)-0.5
        keys0+=t0
        keys1+=t1
        #batch
        batch={
            'R0':torch.from_numpy(R0.astype(np.float32)),
            'R1':torch.from_numpy(R1.astype(np.float32)),
            'keys0':torch.from_numpy(keys0.astype(np.float32)),
            'keys1':torch.from_numpy(keys1.astype(np.float32)),
            'feats0':torch.from_numpy(feats0.astype(np.float32)),
            'feats1':torch.from_numpy(feats1.astype(np.float32)),
            'R':torch.from_numpy(R.astype(np.float32)),
            'true_idx':torch.from_numpy(np.array([R_index]).astype(np.int)),
            'pairs':torch.from_numpy(pairs.astype(np.int)),
            'unpair0':torch.from_numpy(unpair0.astype(np.int)),
            'unpair1':torch.from_numpy(unpair1.astype(np.int)),
            'matches0':torch.from_numpy(matches0.astype(np.int)),
        }
        return batch
    
    def run(self,batchnum=30000,valsize=200):
        train_pcp,val_pcp=self.TV_RM(batchnum)
                    
        for b in tqdm(range(len(val_pcp))[0:valsize]):
            if os.path.exists(f'{self.RMval_dir}/{b}.pth'):continue
            name,pc0,pc1=val_pcp[b]
            batch=self.batch_generator(name,pc0,pc1)
            torch.save(batch,f'{self.RMval_dir}/{b}.pth')
        
        for b in tqdm(range(len(train_pcp))):
            if os.path.exists(f'{self.RMtrain_dir}/{b}.pth'):continue
            name,pc0,pc1=train_pcp[b]
            batch=self.batch_generator(name,pc0,pc1)
            torch.save(batch,f'{self.RMtrain_dir}/{b}.pth')

            

if __name__=="__main__":
    cfg,nouse=get_config()
    Train_creater=RM_trainset(cfg)
    Train_creater.run(batchnum=30000,valsize=200)
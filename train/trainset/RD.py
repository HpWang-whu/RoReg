"""
Trainset preparation for RD and RM.

For data argumentation, we need to add random rotations to scans firstly and keep scan pairs with overlap over 10%.
Already stored in 3dm_train_rot.

RD(YOHO_feature)~score, RM(YOHO_Feature0,YOHO_Feature1)~index pairs. Thus we need the YOHO-Desc of the 3dmatch_train_rot.
    FCGFG(PC)-->YOHO_PartI(PC)-->YOHO_Feature(Save)

For RD:
    Matmul-->pairs(Save)
    YOHO_Feature pairs+YOHO_PartII-->Ts(Save)
"""
import os
import random
import numpy as np
from tqdm import tqdm
from dataops.dataset import get_dataset_name
from parses.parses_train_rd import get_config
from test import name2extractor, name2matcher, extractor_dr_index, extractor_localtrans
from utils.utils import save_pickle


class RD_trainset():
    def __init__(self,cfg):
        self.cfg=cfg
        self.datasets=get_dataset_name(self.cfg.trainset,self.cfg.origin_data_dir)
        self.backbone_w=self.cfg.backbone_w
        self.group=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy')
        self.perms=np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)
        #RD
        self.matcher=name2matcher['matmul'](self.cfg)
        self.extractorI=name2extractor['yoho_des'](self.cfg)
        self.drindex_extractor = extractor_dr_index(self.cfg)
        self.extractorII=extractor_localtrans(self.cfg)
        #save
        self.trainRD_fn=f'{self.cfg.output_cache_fn}/{self.cfg.trainset}/train_RD.pkl'
        self.valRD_fn=f'{self.cfg.output_cache_fn}/{self.cfg.trainset}/val_RD.pkl'
        
    def TV_RD(self):
        train_pcp=[]
        if not os.path.exists(self.trainRD_fn):
            for name,dataset in tqdm(self.datasets.items()):
                if name in ['wholesetname','valscenes']:continue
                if name in self.datasets['valscenes']:continue
                for pair in dataset.pair_ids:
                    pc0,pc1=pair
                    train_pcp.append((name,pc0,pc1))
            save_pickle(train_pcp,self.trainRD_fn)

        val_pcp=[]  
        if not os.path.exists(self.valRD_fn):
            for name in self.datasets['valscenes']:
                dataset=self.datasets[name]
                for pair in dataset.pair_ids:
                    pc0,pc1=pair
                    val_pcp.append((name,pc0,pc1))
            random.shuffle(val_pcp)
            save_pickle(val_pcp,self.valRD_fn)

    def run(self):
        self.TV_RD()
        setname=self.cfg.trainset
        voxelsize=self.cfg.voxelsize
        #generate Fcgf group feature
        os.system(f'python testset.py --model {self.backbone_w} --outdir {self.cfg.output_cache_fn} --voxel_size {voxelsize} --dataset {setname}')
        #pair->coarse->refined
        for name,dataset in tqdm(self.datasets.items()):
            if name in ['wholesetname','valscenes']:continue
            self.extractorI.run(dataset)
            self.matcher.run(dataset,num=self.cfg.keynum)
            self.extractor_dr_index.Rindex(dataset,self.cfg.keynum)
            self.extractorII.Rt_pre(dataset,self.cfg.keynum)
    
if __name__=="__main__":
    cfg,nouse=get_config()
    Train_creater=RD_trainset(cfg)
    Train_creater.TV_RD()
    Train_creater.run()
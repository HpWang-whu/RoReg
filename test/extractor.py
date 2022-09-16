import os,sys
sys.path.append('..')
import time
import torch
import numpy as np
import open3d as o3d
from tqdm import tqdm
from utils.r_eval import matrix_from_quaternion
from torch.utils.data import DataLoader
from utils.utils import transform_points, read_pickle,make_non_exists_dir,to_cuda
from network import name2network

class yoho_des():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network['GF_test'](self.cfg).cuda()
        self.model_fn=f'{self.cfg.model_fn}/GF/model.pth'
        self.best_model_fn=f'{self.cfg.model_fn}/GF/model_best.pth'
        self.test_batch_size = self.cfg.bs_GF

    #Model
    def _load_model(self):
        best_para,start_step=0,0
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            best_para = checkpoint['best_para']
            self.network.load_state_dict(checkpoint['network_state_dict'])
            # print(f'Resuming best para {best_para}')
        else:
            raise ValueError("No model exists")
        
    #Extract
    def run(self,dataset):
        #data input 5000*32*60
        #output: 5000*32*60->save
        self._load_model()
        self.network.eval()
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        FCGF_input_dir=f'{self.cfg.output_cache_fn}/{datasetname}/{self.cfg.backbone}_Input_Group_feature'
        YOHO_output_dir=f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature'
        make_non_exists_dir(YOHO_output_dir)
        print(f'Extracting the PartI descriptors on {dataset.name}')
        for pc_id in tqdm(dataset.pc_ids):
            if os.path.exists(f'{YOHO_output_dir}/{pc_id}.npy'):continue
            Input_feature=np.load(f'{FCGF_input_dir}/{pc_id}.npy') #5000*32*60
            output_feature=[]
            bi=0
            while(bi*self.test_batch_size<Input_feature.shape[0]):
                start=bi*self.test_batch_size
                end=(bi+1)*self.test_batch_size
                batch=torch.from_numpy(Input_feature[start:end,:,:].astype(np.float32)).cuda()
                with torch.no_grad():
                    batch_output=self.network(batch)
                output_feature.append(batch_output['eqv'].detach().cpu().numpy())
                bi+=1
            output_feature=np.concatenate(output_feature,axis=0)
            np.save(f'{YOHO_output_dir}/{pc_id}.npy',output_feature) #5000*32*60

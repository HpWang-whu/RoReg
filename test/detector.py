import os
import torch
import abc
import utils.utils as utils
import time
from tqdm import tqdm
import numpy as np
from network import name2network

class yoho_det():
    def __init__(self,cfg):
        self.cfg=cfg
        self.network=name2network['RD_test'](cfg).cuda()
        self.best_model_fn=f'{self.cfg.model_fn}/RD/model_best.pth'
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self._load_model()

    #Model_import
    def _load_model(self):
        if os.path.exists(self.best_model_fn):
            checkpoint=torch.load(self.best_model_fn)
            self.network.load_state_dict(checkpoint['network_state_dict'],strict=True)
        else:
            raise ValueError("No model exists")
    
    def run(self,dataset):
        self.network.eval()
        if dataset.name[0:4]=='3dLo':
            datasetname=f'3d{dataset.name[4:]}'
        else:
            datasetname=dataset.name
        savedir=f'{self.cfg.output_cache_fn}/{datasetname}/det_score'
        utils.make_non_exists_dir(savedir)
        print(f'Evaluating the saliency of points using rotaion guided detector on {dataset.name}')
        for pc_id in tqdm(range(len(dataset.pc_ids))):
            if os.path.exists(f'{savedir}/{pc_id}.npy'):continue
            feats=np.load(f'{self.cfg.output_cache_fn}/{datasetname}/YOHO_Output_Group_feature/{pc_id}.npy')
            batch={
                'feats':torch.from_numpy(feats.astype(np.float32))
            }
            batch=utils.to_cuda(batch)
            with torch.no_grad():
                scores=self.network(batch)['scores'].cpu().numpy()
            # normalization for NMS comparision only
            argscores = np.argsort(scores)
            scores[argscores] = np.arange(scores.shape[0])/scores.shape[0]
            np.save(f'{savedir}/{pc_id}.npy', scores)

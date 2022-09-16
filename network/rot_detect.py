import torch
import numpy as np
import torch.nn as nn
from network.ops import Residual_Comb_Conv

class detector_eqv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()
        self.eqv_encoder=nn.ModuleList([Residual_Comb_Conv(32, 64, 1, self.Nei_in_SO3)])
        self.det_scorer=nn.Sequential(
            nn.Conv2d(60, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
        )
    
    def forward(self,batch):
        #batch:n*32*60
        feats0=batch['feats0'][0]
        feats1=batch['feats1'][0]
        feat=torch.cat([feats0,feats1],dim=0)#128*32*60
        feat=self.eqv_encoder[0](feat) #b*1*60
        feat=torch.squeeze(feat)[:,:,None,None]
        score=self.det_scorer(feat)[:,0,0,0]
        score=torch.sigmoid(score)
        
        return {
            'scores':score,
            'feats0':torch.mean(feats0,dim=-1),
            'feats1':torch.mean(feats1,dim=-1),
            'Rdiffs':batch['Rdiffs'][0],
            'dist':batch['dist'][0]
        }
            
class detector_eqv_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()
        self.eqv_encoder=nn.ModuleList([Residual_Comb_Conv(32, 64, 1, self.Nei_in_SO3)])
        self.det_scorer=nn.Sequential(
            nn.Conv2d(60, 32, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1, 1),
        )
    
    def forward(self,batch):
        #batch:n*32*60
        feat=batch['feats']
        feat=self.eqv_encoder[0](feat) #b*1*60
        feat=torch.squeeze(feat)[:,:,None,None]
        score=self.det_scorer(feat)[:,0,0,0]
        score=torch.sigmoid(score)
        return {
            'scores':score
        }
              
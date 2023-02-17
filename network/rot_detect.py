import torch
import numpy as np
import torch.nn as nn
from network.ops import Residual_Comb_Conv

# motivated by keypoints developed in 2D domain, we use std rather than sigmoid for a more stable converegence  -- V2
# a similar performance as V1 but easier to train
class detector_eqv(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()
        self.perm=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int).reshape([-1])).cuda()
        self.eqv_encoder=nn.ModuleList([Residual_Comb_Conv(32, 64, 16, self.Nei_in_SO3)])

    def forward(self,batch):
        #batch:n*32*60
        feats0=batch['feats0'][0]
        feats1=batch['feats1'][0]
        feat=torch.cat([feats0,feats1],dim=0)   #128*32*60
        feat=self.eqv_encoder[0](feat)          #128*16*60
        feat=feat/torch.norm(feat,dim=1,keepdim=True)        #128*16*60
        b,f,_ = feat.shape
        #self std
        feat_p=feat[:,:,self.perm].reshape(b,f,60,60)
        feat_std=torch.einsum(f'mfab,mfb->ma',feat_p,feat)
        score=torch.std(feat_std,dim=1)        
        return {
            'scores':score,
            'feats0':torch.mean(feats0,dim=-1),
            'feats1':torch.mean(feats1,dim=-1),
            'Rdiffs':batch['Rdiffs'][0],
        }
            
class detector_eqv_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()
        self.perm=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int).reshape([-1])).cuda()
        self.eqv_encoder=nn.ModuleList([Residual_Comb_Conv(32, 64, 16, self.Nei_in_SO3)])
    
    def forward(self,batch):
        #batch:n*32*60
        feat=batch['feats']
        feat=self.eqv_encoder[0](feat)          #128*16*60
        feat=feat/torch.norm(feat,dim=1,keepdim=True)        #128*16*60
        b,f,_ = feat.shape
        #self std
        feat_p=feat[:,:,self.perm].reshape(b,f,60,60)
        feat_std=torch.einsum(f'mfab,mfb->ma',feat_p,feat)
        score=torch.std(feat_std,dim=1)        
        return {
            'scores':score
        }
              
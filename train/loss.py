# losses of each part
"""
Loss and validations.
"""

import torch
import abc
import utils.utils as utils
import time
from tqdm import tqdm
import numpy as np


class Loss(abc.ABC):
    def __init__(self,keys: list or tuple):
        self.keys=list(keys)

    @abc.abstractmethod
    def __call__(self, data_pr, data_gt, **kwargs):
        pass
    
#loss group feature extractor
class Batch_hard_Rindex_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['Batch_hard_Rindex_loss'])
        self.R_perm=torch.from_numpy(np.load(f'{cfg.SO3_related_files}/60_60.npy').astype(np.int).reshape([-1])).cuda()
        self.class_loss=torch.nn.CrossEntropyLoss()

    def eqvloss(self,eqvfeat0,eqvfeat1):
        B,F,G=eqvfeat0.shape
        eqvfeat0=eqvfeat0[:,:,self.R_perm].reshape([B,F,G,G])
        score=torch.einsum('bfgk,bfk->bg',eqvfeat0,eqvfeat1)
        return score

    def __call__(self, data_pr, batch):
        Index= data_pr['DR_true_index'].type(torch.int64)
        
        B=Index.shape[0]
        feats0=data_pr['feats0_inv'] # bn,f
        feats1=data_pr['feats1_inv'] # bn,f

        B,L = feats1.shape
        q_vec=feats0.contiguous().view(B, 1, L)
        ans_vecs=feats1.contiguous().view(1, B, L)
        dist = ((q_vec - ans_vecs) ** 2).sum(-1)
        dist = torch.nn.functional.log_softmax(dist, 1)
        loss_true = torch.diag(dist)
        loss_false=torch.min(dist+torch.eye(B).cuda(),dim=1)[0]
        loss=torch.mean(torch.clamp_min(loss_true-loss_false+0.3,0))

        score=self.eqvloss(data_pr['feats0_eqv_af_conv'],data_pr['feats1_eqv_af_conv'])
        eqv_loss=self.class_loss(score,Index)
        return 5*loss+eqv_loss

# loss for rotation guided detector
class RD_loss(Loss):
    #coarse/precision?
    def __init__(self,cfg):
        super().__init__(['RD_loss'])

    def __call__(self,output, batch):
        '''
        output['scores']:2n
        output['Rdiffs']:n
        '''
        feats0=output['feats0'] #b*f
        feats1=output['feats1'] #b*f
        Rdiffs=output['Rdiffs'] #b
        scores=output['scores'] #b

        bs=Rdiffs.shape[0]
        scoresA=scores[0:bs]
        scoresB=scores[bs:]
        scores=scoresA+scoresB
        scores=scores/torch.mean(scores)
        loss=torch.mean(scores*Rdiffs)
        return loss

# loss for rotation coherence detector
class RM_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['RM_loss'])
    
    def ds(self,scores,pairs):
        ploss=0
        uloss=0
        eps=1e-5
        for i in range(scores.shape[0]):
            logscore=-torch.log(scores[i,:,:]+eps)
            ploss+=torch.mean(logscore[pairs[i,:,0],pairs[i,:,1]])
        return ploss    
    
    def ot(self,output,batch):
        ploss=0
        uloss=0
        scores=output['scores']
        pairs=batch['pairs']
        unpair0=batch['unpair0']
        unpair1=batch['unpair1']
        for i in range(scores.shape[0]):
            logscore=-scores[i,:,:]
            ploss+=torch.mean(logscore[pairs[i,:,0],pairs[i,:,1]])
            uloss+=torch.mean(logscore[unpair0[i],-1])+torch.mean(logscore[-1,unpair1[i]])
        return ploss+uloss
    
    def __call__(self,output,batch):
        scores_other=output['scores_other']
        pairs=batch['pairs']
        ol=0
        olnum=scores_other.shape[-1]
        for l in range(olnum):
            ol+=self.ds(scores_other[:,:,:,l],pairs)
            
        ll=self.ot(output,batch)
        return ol+olnum*ll


#loss for rotation estimation block
class L1_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['L1_Loss'])
        self.loss=torch.nn.SmoothL1Loss(reduction='sum')
    
    def __call__(self,output,batch):
        patch_gt = torch.squeeze(batch['deltaR'])
        patch_op = torch.squeeze(output['quaternion_pre'])
        return self.loss(patch_op,patch_gt)


class L2_loss(Loss):
    def __init__(self,cfg):
        super().__init__(['L2_Loss'])
        self.loss=torch.nn.MSELoss(reduction='sum')
    
    def __call__(self,output,batch):
        patch_gt = torch.squeeze(batch['deltaR'])
        patch_op = torch.squeeze(output['quaternion_pre'])
        return self.loss(patch_op,patch_gt)


name2loss={
    'loss_gf':Batch_hard_Rindex_loss,
    'loss_rd':RD_loss,
    'loss_rm':RM_loss,
    'loss_et':L2_loss
}
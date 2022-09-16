import torch
import torch.nn as nn
import numpy as np
from network.ops import *


class Group_feat_network(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()

        self.Conv_in=nn.Sequential(nn.Conv2d(32,256,(1,13),1))
        self.SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        self.Conv_out=Comb_Conv(256,32)

    def data_process(self,data):
        data=torch.squeeze(data)
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_in(data)[:,:,:,0]
        for layer in range(len(self.SO3_Conv_layers)):
            data=self.SO3_Conv_layers[layer](data)
        data=self.data_process(data)
        data=self.Conv_out(data)[:,:,:,0]
        return data

        
    def forward(self, feats):
        feats_eqv=self.SO3_Conv(feats)# bn,f,gn
        feats_eqv=feats_eqv+feats
        feats_inv=torch.mean(feats_eqv,dim=-1)# bn,f

        #before conv for partII
        feats_eqv=feats_eqv/torch.clamp_min(torch.norm(feats_eqv,dim=1,keepdim=True),min=1e-4)
        feats_inv=feats_inv/torch.clamp_min(torch.norm(feats_inv,dim=1,keepdim=True),min=1e-4)

        return {'inv':feats_inv,'eqv':feats_eqv}

class GF_train(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.PartI_net=Group_feat_network(self.cfg)
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 
        
    
    def Des2DR(self,Des1,Des2):#before_rot after_rot
        Des1=Des1[:,:,torch.reshape(self.R_index_permu,[-1])].reshape([Des1.shape[0],Des1.shape[1],60,60])
        cor=torch.einsum('bfag,bfg->ba',Des1,Des2)
        return torch.argmax(cor,dim=1)

    def forward(self,data):
        feats0=torch.squeeze(data['feats0']) # bn,32,60
        feats1=torch.squeeze(data['feats1']) # bn,32,60
        true_idxs=torch.squeeze(data['true_idx']) # bn
        yoho_0=self.PartI_net(feats0)
        yoho_1=self.PartI_net(feats1)
        pre_idxs=self.Des2DR(yoho_0['eqv'],yoho_1['eqv'])
        #pre_idxs=self.Des2DR(feats0,feats1)
        part1_ability=torch.mean((pre_idxs==true_idxs).type(torch.float32))

        return {'feats0_eqv_bf_conv':feats0,
                'feats1_eqv_bf_conv':feats1,
                'feats0_eqv_af_conv':yoho_0['eqv'],
                'feats1_eqv_af_conv':yoho_1['eqv'],
                'feats0_inv':yoho_0['inv'],
                'feats1_inv':yoho_1['inv'],
                'DR_pre_ability':part1_ability, # no use for partI
                'DR_true_index':true_idxs,
                'DR_pre_index':pre_idxs}        # no use for partI

class GF_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg
        self.PartI_net=Group_feat_network(self.cfg)

    def forward(self,group_feat):
        return self.PartI_net(group_feat)
    
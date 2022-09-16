import torch
import torch.nn as nn
import numpy as np
from network.ops import *
from network.group_feat import GF_train

class ET_train(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 


        self.PartI_net=GF_train(self.cfg)
        self.Conv_init=Comb_Conv(32*4,256)
        self.PartII_SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        
        self.PartII_To_R_dims=[256,512,128,4]
        self.PartII_To_R_FC=nn.Sequential(
            nn.Conv2d(self.PartII_To_R_dims[0],self.PartII_To_R_dims[1],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[1]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[1],self.PartII_To_R_dims[2],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[2]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[2],self.PartII_To_R_dims[3],1,1)
        )
        
    def data_process(self,data):
        data=torch.squeeze(data)
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def PartII_SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_init(data)[:,:,:,0]
        for layer in self.PartII_SO3_Conv_layers:
            data=layer(data)
        return data #data:bn,f,gn
    
    def forward(self, data):
        true_idxs=torch.squeeze(data['true_idx']) # bn
        self.PartI_net.eval()
        with torch.no_grad():
            PartI_output=self.PartI_net(data)
        feats0=PartI_output['feats0_eqv_bf_conv'].detach()
        feats1=PartI_output['feats1_eqv_bf_conv'].detach()
        feats0_eqv=PartI_output['feats0_eqv_af_conv'].detach()
        feats1_eqv=PartI_output['feats1_eqv_af_conv'].detach()
        part1_ability=PartI_output['DR_pre_ability'].detach()
        pre_idxs=PartI_output['DR_pre_index'].detach()
        for i in range(feats0.shape[0]):
            feats0[i]=feats0[i,:,self.R_index_permu[true_idxs[i]]]
            feats0_eqv[i]=feats0_eqv[i,:,self.R_index_permu[true_idxs[i]]]
        feats_eqv=torch.cat([feats0,feats1,feats0_eqv,feats1_eqv],dim=1)
        
        feats_eqv=self.PartII_SO3_Conv(feats_eqv)#bn f gn

        feats_inv=torch.mean(feats_eqv,dim=-1)
        feats_inv=feats_eqv.unsqueeze(-1)#bn f 1 1
        feats_inv=feats_eqv.unsqueeze(-1)
        feats_inv=self.PartII_To_R_FC(feats_inv)#bn 4 1 1
        quaternion_pre=feats_inv[:,:,0,0]
        #quaternion_pre=quaternion_pre/torch.norm(quaternion_pre,dim=1)[:,None]
        
        return {'quaternion_pre':quaternion_pre,
                'part1_ability':part1_ability,
                'pre_idxs':pre_idxs,
                'true_idxs':true_idxs}



class ET_test(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg=cfg

        self.Rgroup_npy=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy').astype(np.float32)
        self.Rgroup=torch.from_numpy(self.Rgroup_npy).cuda()
        self.Nei_in_SO3=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/Nei_Index_in_SO3_ordered_13.npy').astype(np.int).reshape([-1])).cuda()    #nei 60*12 readin
        self.R_index_permu=torch.from_numpy(np.load(f'{self.cfg.SO3_related_files}/60_60.npy').astype(np.int)).cuda() 

        self.Conv_init=Comb_Conv(32*4,256)
        self.PartII_SO3_Conv_layers=nn.ModuleList([Residual_Comb_Conv(256,512,256,self.Nei_in_SO3)])
        
        self.PartII_To_R_dims=[256,512,128,4]
        self.PartII_To_R_FC=nn.Sequential(
            nn.Conv2d(self.PartII_To_R_dims[0],self.PartII_To_R_dims[1],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[1]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[1],self.PartII_To_R_dims[2],1,1),
            nn.BatchNorm2d(self.PartII_To_R_dims[2]),
            nn.ReLU(),
            nn.Conv2d(self.PartII_To_R_dims[2],self.PartII_To_R_dims[3],1,1)
        )
        

    def data_process(self,data):
        data=torch.squeeze(data)
        if len(data.size())==2:
            data=data[None,:,:]
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def PartII_SO3_Conv(self,data):#data:bn,f,gn
        data=self.data_process(data)
        data=self.Conv_init(data)[:,:,:,0]
        for layer in self.PartII_SO3_Conv_layers:
            data=layer(data)
        return data #data:bn,f,gn
    
    
    def forward(self, data):
        feats0_eqv_bf_conv=data['before_eqv0']
        feats1_eqv_bf_conv=data['before_eqv1']
        feats0_eqv_af_conv=data['after_eqv0']
        feats1_eqv_af_conv=data['after_eqv1']
        pre_idxs=data['pre_idx']
        
        for i in range(feats0_eqv_bf_conv.shape[0]):
            feats0_eqv_bf_conv[i]=feats0_eqv_bf_conv[i,:,self.R_index_permu[pre_idxs[i]]]
            feats0_eqv_af_conv[i]=feats0_eqv_af_conv[i,:,self.R_index_permu[pre_idxs[i]]]
        feats_eqv=torch.cat([feats0_eqv_bf_conv,feats1_eqv_bf_conv,feats0_eqv_af_conv,feats1_eqv_af_conv],dim=1)

        feats_eqv=self.PartII_SO3_Conv(feats_eqv)#bn f gn
        feats_inv=torch.mean(feats_eqv,dim=-1)#bn f
        feats_inv=feats_eqv.unsqueeze(-1)#bn f 1 1
        feats_inv=feats_eqv.unsqueeze(-1)
        feats_inv=self.PartII_To_R_FC(feats_inv)#bn 4 1 1
        quaternion_pre=feats_inv[:,:,0,0]
        quaternion_pre=quaternion_pre/torch.norm(quaternion_pre,dim=1)[:,None]
        return {'quaternion_pre':quaternion_pre,'pre_idxs':pre_idxs}

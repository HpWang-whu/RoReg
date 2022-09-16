"""
Model in Pytorch of YOHO.
"""

import torch
import torch.nn as nn
import numpy as np


#DRnet
class Comb_Conv(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__()
        self.comb_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
        )
    def forward(self,input):
        return self.comb_layer(input)

class Residual_Comb_Conv(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim,Nei_in_SO3):
        super().__init__()
        self.Nei_in_SO3=Nei_in_SO3
        self.comb_layer_in=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,middle_dim,(1,13),1)
        )
        self.comb_layer_out=nn.Sequential(
            nn.BatchNorm2d(middle_dim),
            nn.ReLU(),
            nn.Conv2d(middle_dim,out_dim,(1,13),1)
        )
        self.short_cut=False
        if not in_dim==out_dim:
            self.short_cut=True
            self.short_cut_layer=nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(),
            nn.Conv2d(in_dim,out_dim,(1,13),1)
            )
    
    def data_process(self,data):
        data=torch.squeeze(data)
        if len(data.size())==2:
            data=data[None,:,:]
        data=data[:,:,self.Nei_in_SO3]
        data=torch.reshape(data,[data.shape[0],data.shape[1],60,13])
        return data

    def forward(self,feat_input):#feat:bn*f*60
        feat=self.data_process(feat_input)
        feat=self.comb_layer_in(feat)
        feat=self.data_process(feat)
        feat=self.comb_layer_out(feat)[:,:,:,0]
        if self.short_cut:
            feat_sc=self.data_process(feat_input)
            feat_sc=self.short_cut_layer(feat_sc)[:,:,:,0]
        else:
            feat_sc=feat_input
        
        return feat+feat_sc #output:bn*f*60
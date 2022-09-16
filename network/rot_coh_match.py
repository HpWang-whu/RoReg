import numpy as np
from copy import deepcopy
import torch
import math
import torch.nn as nn


def score_mat(source,target):
    #source:bn*fn*m*1
    #target:bn*fn*n*1
    score=torch.einsum('bfmo,bfno->bmn',source,target)
    return score

class mlp_2layer(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_dim,middle_dim,1,1),
            nn.InstanceNorm2d(middle_dim),
            nn.ReLU(),
            nn.Conv2d(middle_dim,out_dim,1,1)
        )
        self.res_sign=False
        if not in_dim==out_dim:
            self.res=nn.Conv2d(in_dim,out_dim,1,1)
            self.res_sign=True
    def forward(self,inputs):
        #input: BN*in_dim*w*h
        out=self.net(inputs)
        if self.res_sign:
            out+=self.res(inputs)
        return out

def Knn_index_extract(score_mat,k,axis):
    #score_mat: bn*(m*n)
    #k:sample num
    #axis=1 :return bn*(n*k)-->for b*f*n*k, axis=2:return bn*(m*k)-->for b*f*m*k
    argsorts=torch.argsort(score_mat,axis=axis,descending=True) # bn*(m*n)
    return_indics=[]
    if axis==1:
        return argsorts[:,0:k,:].permute(0,2,1)
    elif axis==2:
        return argsorts[:,:,0:k]
    else:
        print('wrong sign')


def Knn_feat_extract(feat,indics):
    #if indics's axis=1 feat should be b*f*m*1, indics should be b*n*k --> b*f*n*k
    #if indics's axis=2 feat should be b*f*n*1, indics should be b*m*k --> b*f*m*k
    device = feat.device
    feat=feat[:,:,:,0].permute(0,2,1) #b*m*f
    B = feat.shape[0]
    view_shape = list(indics.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(indics.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    feat_knn = feat[batch_indices, indics, :].permute(0,3,1,2)
    return feat_knn
 

class Contextnorm(nn.Module):
    def __init__(self,in_dim,middle_dim,out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(in_dim,middle_dim,1,1),
            nn.InstanceNorm2d(middle_dim),
            nn.ReLU(),
            nn.Conv2d(middle_dim,out_dim,1,1)
        )
        self.res_sign=False
        if not in_dim==out_dim:
            self.res=nn.Conv2d(in_dim,out_dim,1,1)
            self.res_sign=True
    def forward(self,inputs):
        #input: BN*in_dim*w*h
        out=self.net(inputs)
        if self.res_sign:
            out+=self.res(inputs)
        return out


def attention(query, key, value):
    #query  bn*f'n*hn*m
    #key    bn*f'n*hn*m*k
    #value  bn*f'n*hn*m*k
    #return bn*f'n*hn*m
    dim = query.shape[1]
    scores = torch.einsum('bfhm,bfhmk->bhmk', query, key) / dim**.5
    prob = torch.nn.functional.softmax(scores, dim=-1)
    return torch.einsum('bhmk,bdhmk->bdhm', prob, value), torch.sum(prob,dim=1).reshape(prob.shape[0],-1,prob.shape[-1])


class MultiHeadedAttention(nn.Module):
    """ Multi-head attention to increase model expressivitiy """
    def __init__(self, num_heads: int, d_model: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.dim = d_model // num_heads
        self.num_heads = num_heads
        self.merge = nn.Conv2d(d_model, d_model, kernel_size=1,stride=1)
        self.proj = nn.ModuleList([deepcopy(self.merge) for _ in range(3)])

    def forward(self, query, key, value):
        #query  bn*fn*m*1
        #key    bn*fn*m*k
        #value  bn*fn*m*k
        #return bn*fn*m*1
        batch_dim,feat_dim,point_num,kn=key.shape
        query, key, value = [l(x).view(batch_dim, self.dim, self.num_heads, -1)
                             for l, x in zip(self.proj, (query, key, value))]
        #query  bn*f'n*hn*(m*1)
        #key    bn*f'n*hn*(m*k)
        #value  bn*f'n*hn*(m*k)
        key = key.reshape(batch_dim,self.dim,self.num_heads,point_num,kn)
        value = value.reshape(batch_dim,self.dim,self.num_heads,point_num,kn)
        x, prob = attention(query, key, value)
        return self.merge(x.contiguous().view(batch_dim, self.dim*self.num_heads, point_num, -1)),prob


class Cross_attention_block(nn.Module):
    def __init__(self,cross_k,s2t:bool):
        super().__init__()
        self.k=cross_k
        self.s2t=s2t
        self.perms=torch.from_numpy(np.load(f'{groupdir}/60_60.npy').astype(np.int)).cuda().reshape(-1)
        self.cross_attn=MultiHeadedAttention(4,32)
        self.merge=mlp_2layer(32*3,64,32)

    
    def forward(self,source,target,source_eqv,target_eqv,featinv):
        #source:b*f*m*1
        #target:b*f*n*1
        #source_eqv:b*f*m*g
        #target_eqv:b*f*n*g
        #featinv:b*f*m*1
        #return:b*f*m*1,b*g*m*1

        #knn extraction in the feature space for the possible correspondence
        #score mat
        score=score_mat(source,target) #bn*m*n
        #knn extract
        knn_ind=Knn_index_extract(score,self.k,axis=2)  #b*m*k
        nn_ind=Knn_index_extract(score,1,axis=2)        #b*m*1
        knn_fea=Knn_feat_extract(target,knn_ind)        #b*f*m*k

        #feat similarity+enhancement
        feat_out,_=self.cross_attn(source,knn_fea,knn_fea)
        feat_out=self.merge(torch.cat([featinv,source,feat_out],dim=1))

        #coarse rotaion indicator
        #max sim with dimension b*m in b*m*k
        b,f,n,g=target_eqv.shape
        #extractor of the nn one
        target_eqv=Knn_feat_extract(target_eqv.permute(0,1,3,2).reshape(b,-1,n,1),nn_ind)#b*(f*g)*m*1
        target_eqv=target_eqv.reshape(b,f,g,-1)
        if self.s2t:
            source_eqv=source_eqv[:,:,:,self.perms].reshape(b,f,-1,g,g)
            R_indicator=torch.einsum('bfghm,bfgm->bhm',source_eqv.permute(0,1,3,4,2),target_eqv)[:,:,:,None]
        else:
            target_eqv=target_eqv[:,:,self.perms,:].reshape(b,f,g,g,-1)
            R_indicator=torch.einsum('bfghn,bfgn->bhn',target_eqv,source_eqv.permute(0,1,3,2))[:,:,:,None]
        
        return feat_out,R_indicator



class Self_attention_block(nn.Module):
    '''
    self attention block should encode:
    (1)local feature pattern            from b*f*m*1+b*f*m*k
    (2)orientation consistance          from b*g*m*1
    (3)feature spatial distribution     from b*c*m*1+b*c*m*k
    '''
    def __init__(self,self_k,source:bool):
        super().__init__()
        self.k=self_k
        self.source=source
        self.Rs=torch.from_numpy(np.load(f'{groupdir}/Rotation.npy').astype(np.float32)).cuda()
        self.self_attn=MultiHeadedAttention(4,32)
        self.pos_en=mlp_2layer(3,64,32)
        self.ambiguity=Contextnorm(120,128,32)
        self.val_en=mlp_2layer(32*3,64,32)
        self.merge=mlp_2layer(32*3,64,32)

    def forward(self,feat,coor,R_indicator,featinv):
        #R_indicator exists only when source self attention
        #score mat
        score=score_mat(feat,feat)
        #knn extract
        knn_ind=Knn_index_extract(score,self.k,axis=2)
        knn_fea=Knn_feat_extract(feat,knn_ind)      #b*f*m*k
        knn_coor=Knn_feat_extract(coor,knn_ind)     #b*c*m*k
        knn_coor=knn_coor-coor                      #b*c*m*k
        
        #value carrying the co location information b*f*m*k
        knn_coor=self.pos_en(knn_coor)
        #confidence
        b,g,m,_=R_indicator.shape
        R_indicator=torch.cat([R_indicator,torch.max(R_indicator,dim=2,keepdim=True)[0].repeat(1,1,m,1)],dim=1)
        conf=self.ambiguity(R_indicator)
        knn_coor=knn_coor/torch.norm(knn_coor,2,1,keepdim=True)
        knn_fea=knn_fea/torch.norm(knn_fea,2,1,keepdim=True)
        conf=conf/torch.norm(conf,2,1,keepdim=True)
        value=self.val_en(torch.cat([knn_coor,knn_fea,conf.repeat(1,1,1,self.k)],dim=1))

        feat_out,_=self.self_attn(feat,knn_fea,value)
        feat_out=self.merge(torch.cat([featinv,feat,feat_out],dim=1))
        return feat_out



class Merge_info_block(nn.Module):
    def __init__(self,self_k,cross_k):
        super().__init__()
        self.cross_graph_s2t=Cross_attention_block(cross_k,s2t=True)
        self.self_graph_s=Self_attention_block(self_k,source=True)
        self.cross_graph_t2s=Cross_attention_block(cross_k,s2t=False)
        self.self_graph_t=Self_attention_block(self_k,source=False)


    def forward(self,source,target,source_eqv,target_eqv,source_coor,target_coor,source_inv,target_inv):
        #source:bn*fn*m*1
        #target:bn*fn*n*1
        #source_eqv:bn*fn*m*g
        #target_eqv:bn*fn*n*g
        #source_coor:bn*3*m*1
        #target_coor:bn*3*n*1
        #source_inv:bn*fn*m*1
        #target_inv:bn*fn*n*1
        #return
        #eh_source:bn*fn*m*1
        #eh_target:bn*fn*n*1
        #R_ind_s2t:bn*60*m*1
        #R_ind_t2s:bn*60*n*1
        source_s2t,R_ind_s2t=self.cross_graph_s2t(source,target,source_eqv,target_eqv,source_inv)
        eh_source=self.self_graph_s(source_s2t,source_coor,R_ind_s2t,source_inv)
        target_t2s,R_ind_t2s=self.cross_graph_t2s(target,source,target_eqv,source_eqv,target_inv)
        eh_target=self.self_graph_t(target_t2s,target_coor,R_ind_t2s,target_inv)
        return eh_source,eh_target
    

class Graph_enhance_net(nn.Module):
    def __init__(self):
        super().__init__()
        self.merge_blocks=nn.ModuleList(
            [Merge_info_block(16,16),
            Merge_info_block(8,8)]
            )
    
    def forward(self,source_eqv,target_eqv,source_coor,target_coor,source_inv,target_inv):
        #source_eqv:b*32*m*g
        #target_eqv:b*32*n*g
        #source_coor:b*3*m
        #target_coor:b*3*n
        #source_inv:bn*fn*m*1
        #target_inv:bn*fn*n*1
        #return
        #source:bn*fn*m*layer
        #target:bn*fn*n*layer

        sources=[]
        targets=[]
        source=torch.mean(source_eqv,dim=-1)[:,:,:,None]
        target=torch.mean(target_eqv,dim=-1)[:,:,:,None]
        
        for layer in self.merge_blocks:
            source,target=layer(source,target,source_eqv,target_eqv,source_coor,target_coor,source_inv,target_inv)
            sources.append(source)
            targets.append(target)
        source=torch.cat(sources,dim=-1)
        target=torch.cat(targets,dim=-1)
        return source,target #b*g*m*4


class sinkhorn_ot(nn.Module):
    def __init__(self,origin_bin,iters):
        super().__init__()
        self.iters=iters
        bin_score = torch.nn.Parameter(torch.tensor(origin_bin))
        self.register_parameter('bin_score', bin_score)
    

    def log_sinkhorn_iterations(self, Z, log_mu, log_nu, iters: int):
        """ Perform Sinkhorn Normalization in Log-space for stability"""
        u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
        for _ in range(iters):
            u = log_mu - torch.logsumexp(Z + v.unsqueeze(1), dim=2)
            v = log_nu - torch.logsumexp(Z + u.unsqueeze(2), dim=1)
        return Z + u.unsqueeze(2) + v.unsqueeze(1)
    

    def log_optimal_transport(self, scores, alpha, iters: int):
        """ Perform Differentiable Optimal Transport in Log-space for stability"""
        b, m, n = scores.shape
        one = scores.new_tensor(1)
        ms, ns = (m*one).to(scores), (n*one).to(scores)

        bins0 = alpha.expand(b, m, 1)
        bins1 = alpha.expand(b, 1, n)
        alpha = alpha.expand(b, 1, 1)

        couplings = torch.cat([torch.cat([scores, bins0], -1),
                            torch.cat([bins1, alpha], -1)], 1)

        norm = - (ms + ns).log()
        log_mu = torch.cat([norm.expand(m), ns.log()[None] + norm])
        log_nu = torch.cat([norm.expand(n), ms.log()[None] + norm])
        log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)

        Z = self.log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
        Z = Z - norm  # multiply probabilities by M+N
        return Z


    def forward(self,scores):
        #scores:bn(1)*k1n*k2n
        return self.log_optimal_transport(scores,self.bin_score,self.iters)


groupdir = './utils/group_related'
class Match_ot(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        self.cfg=cfg
        groupdir = self.cfg.SO3_related_files
        self.coor_norm_step=0.025
        self.Graph=Graph_enhance_net()
        self.perms=torch.from_numpy(np.load(f'{groupdir}/60_60.npy').astype(np.int)).cuda().reshape(-1)
        self.final_mlp=mlp_2layer(64,64,32)
        self.ot_layer=sinkhorn_ot(0.2,100)


    def arange_like(self,x, dim: int):
        return x.new_ones(x.shape[dim]).cumsum(0) - 1  # traceable in 1.1
    
    
    def forward(self,batch):
        source_eqv=batch['feats0'].permute(0,2,1,3)
        target_eqv=batch['feats1'].permute(0,2,1,3)
        source_coor=batch['keys0'].permute(0,2,1)[:,:,:,None]/self.coor_norm_step
        target_coor=batch['keys1'].permute(0,2,1)[:,:,:,None]/self.coor_norm_step

        #score inv
        source_inv=torch.mean(source_eqv,dim=-1,keepdim=True)
        target_inv=torch.mean(target_eqv,dim=-1,keepdim=True)
        source,target=self.Graph(source_eqv,target_eqv,source_coor,target_coor,source_inv,target_inv)
        #source:bn*fn*m*layer
        #target:bn*fn*n*layer
        #R_ind_s2ts:bn*60*m*layer
        #R_ind_t2ss:bn*60*n*layer

        # bn*m*n*(layer-1)
        scores_otherlayer=torch.einsum('bfmo,bfno->bmno',source,target)
        scores_otherlayer0=nn.functional.softmax(scores_otherlayer,dim=-3)
        scores_otherlayer1=nn.functional.softmax(scores_otherlayer,dim=-2)
        scores_other=scores_otherlayer0*scores_otherlayer1
        
        #final layer supervision
        source_final=self.final_mlp(torch.cat([source_inv,source[:,:,:,-1:]],dim=1))
        target_final=self.final_mlp(torch.cat([target_inv,target[:,:,:,-1:]],dim=1))
        score=torch.einsum('bfmo,bfno->bmn',source_final,target_final)

        #final layer optimal transport
        scores_bin=self.ot_layer(score)

        #matmul correspondence prediction
        max0, max1 = scores_bin[:, :-1, :-1].max(2), scores_bin[:, :-1, :-1].max(1)
        indices0, indices1 = max0.indices, max1.indices
        mutual0 = self.arange_like(indices0, 1)[None] == indices1.gather(1, indices0)
        mutual1 = self.arange_like(indices1, 1)[None] == indices0.gather(1, indices1)
        zero = scores_bin.new_tensor(0)
        mscores0 = torch.where(mutual0, max0.values.exp(), zero)
        mscores1 = torch.where(mutual1, mscores0.gather(1, indices1), zero)
        valid0 = mutual0
        valid1 = mutual1 & valid0.gather(1, indices1)
        indices0 = torch.where(valid0, indices0, indices0.new_tensor(-1))
        indices1 = torch.where(valid1, indices1, indices1.new_tensor(-1))    

        return {
            'scores':scores_bin,      # b*(m+1,n+1)
            'scores_other':scores_other, # b*m*n*(layer-1)
            'matches0': indices0, # use -1 for invalid match
            'matches1': indices1, # use -1 for invalid match
            'matching_scores0': mscores0,
            'matching_scores1': mscores1,
            'source_final':source_final,
            'target_final':target_final
        }
    
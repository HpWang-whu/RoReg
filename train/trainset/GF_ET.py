"""
Generate Trainset using 3dmatch_train for PartI and PartII.
PC*60 rotations->FCGF backbone-> FCGF Group feature for PC keypoints;
PC + PCA filter -->new keys for less training noise;
PC pair + gt --> gt pps;
pps + FCGF Group feature --> batch.
"""

import os
import numpy as np
import open3d as o3d
import torch
import random
from tqdm import tqdm
from parses.parses_train_gf import get_config
from utils.r_eval import compute_R_diff,quaternion_from_matrix
from dataops.dataset import get_dataset_name
from utils.utils import make_non_exists_dir,random_rotation_matrix,read_pickle,save_pickle, transform_points
from backbone.fcgf.misc import extract_features
from backbone.fcgf import load_model


class GF_ET_trainset():
    def __init__(self,cfg):
        self.cfg = cfg
        self.trainset= self.cfg.trainset
        self.origin_data_dir=self.cfg.origin_data_dir
        self.datasets=get_dataset_name(self.trainset,self.origin_data_dir)
        self.output_dir= self.cfg.output_cache_fn
        self.Rgroup=np.load(f'{self.cfg.SO3_related_files}/Rotation.npy')
        self.valscenes=self.datasets['valscenes']
        self.Rnum = self.cfg.Rnum

    def PCA_keys_sample(self):
        for name,dataset in tqdm(self.datasets.items()):
            if name in ['wholesetname','valscenes']:continue

            Save_keys_dir=f'{self.output_dir}/Filtered_Keys/{dataset.name}'
            Save_pair_dir=f'{self.output_dir}/Pairs/{dataset.name}'
            make_non_exists_dir(Save_keys_dir)
            make_non_exists_dir(Save_pair_dir)

            for pc_id in tqdm(dataset.pc_ids): #index in pc
                if os.path.exists(f'{Save_keys_dir}/{pc_id}_index.npy'):continue
                Keys_index=np.loadtxt(dataset.get_key_dir(pc_id)).astype(np.int)
                Keys=dataset.get_kps(pc_id)
                Pcas=np.load(f'{dataset.root}/pca_0.3/{pc_id}.npy')
                Ok_index=np.arange(Pcas.shape[0])[Pcas[:,0]>self.cfg.pca_thre].astype(np.int)
                Keys=Keys[Ok_index]
                Keys_index=Keys_index[Ok_index]
                #Save the filtered index
                np.save(f'{Save_keys_dir}/{pc_id}_coor.npy',Keys)
                np.save(f'{Save_keys_dir}/{pc_id}_index.npy',Keys_index) #in pc
            
            #pair with the filtered keypoints: index in keys
            for pair in tqdm(dataset.pair_ids):
                pc0,pc1=pair
                if os.path.exists(f'{Save_pair_dir}/{pc0}-{pc1}.npy'):continue
                keys0=torch.from_numpy(np.load(f'{Save_keys_dir}/{pc0}_coor.npy').astype(np.float32)).cuda()
                keys1=np.load(f'{Save_keys_dir}/{pc1}_coor.npy')
                gt = dataset.get_transform(pc0,pc1)
                keys1 = transform_points(keys1, gt)
                keys1 = torch.from_numpy(keys1.astype(np.float32)).cuda()
                diff=torch.norm(keys0[:,None,:]-keys1[None,:,:],dim=-1).cpu().numpy()
                pair=np.where(diff<self.cfg.pps_thre)
                pair=np.concatenate([pair[0][:,None],pair[1][:,None]],axis=1)# pairnum*2
                np.save(f'{Save_pair_dir}/{pc0}-{pc1}.npy',pair)

    
    def FCGF_Group_Feature_Extractor(self,Point,Keys_index): #index in pc
        #output:kn*32*60
        output=[]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(self.cfg.backbone_w)
        config = checkpoint['config']

        num_feats = 1
        Model = load_model(config.model)
        model = Model(
            num_feats,
            config.model_n_out,
            bn_momentum=0.05,
            normalize_feature=config.normalize_feature,
            conv1_kernel_size=config.conv1_kernel_size,
            D=3)
            
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        model = model.to(device)
        
        for i in range(self.Rgroup.shape[0]):
            one_R_output=[]
            R_i=self.Rgroup[i]
            Point_i=Point@R_i.T
            Keys_i=Point_i[Keys_index]
            with torch.no_grad():
                xyz_down, feature = extract_features(
                                    model,
                                    xyz=Point_i,
                                    voxel_size=self.cfg.voxelsize,
                                    device=device,
                                    skip_check=True)
            feature=feature.cpu().numpy()
            xyz_down_pcd = o3d.geometry.PointCloud()
            xyz_down_pcd.points = o3d.utility.Vector3dVector(xyz_down)
            pcd_tree = o3d.geometry.KDTreeFlann(xyz_down_pcd)
            for k in range(Keys_i.shape[0]):
                [_, idx, _] = pcd_tree.search_knn_vector_3d(Keys_i[k], 1)
                one_R_output.append(feature[idx[0]][None,:])
            one_R_output=np.concatenate(one_R_output,axis=0)#kn*32
            output.append(one_R_output[:,:,None])
        return np.concatenate(output,axis=-1) #kn*32*60


    def PC_random_rot_feat(self):
        for key,dataset in tqdm(self.datasets.items()):
            if key in ['wholesetname','valscenes']:continue
            for pc_id in tqdm(dataset.pc_ids):
                Feats_save_dir=f'{self.output_dir}/Rotated_Features/{dataset.name}'
                make_non_exists_dir(Feats_save_dir)
                if os.path.exists(f'{Feats_save_dir}/{pc_id}_feats.npz'):continue
                Random_Rs=[]
                Feats=[]

                PC=dataset.get_pc(pc_id)
                Key_idx=np.load(f'{self.output_dir}/Filtered_Keys/{dataset.name}/{pc_id}_index.npy')
                
                for R_i in range(self.Rnum):
                    R_one=random_rotation_matrix()
                    Random_Rs.append(R_one[None,:,:])
                Random_Rs=np.concatenate(Random_Rs,axis=0)# 5*3*3

                for R_i in range(self.Rnum):
                    PC_one=PC@Random_Rs[R_i].T
                    feat_one=self.FCGF_Group_Feature_Extractor(PC_one,Key_idx) #kn*32*60
                    Feats.append(feat_one[None,:,:,:])
                Feats=np.concatenate(Feats,axis=0)#5*kn*32*60

                np.save(f'{Feats_save_dir}/{pc_id}_Rs.npy',Random_Rs)
                np.savez(f'{Feats_save_dir}/{pc_id}_feats.npz',Rs=Random_Rs,feats=Feats)
    

    def R2DR_id(self,R):
        min_diff=180
        best_id=0
        for R_id in range(self.Rgroup.shape[0]):
            R_diff=compute_R_diff(self.Rgroup[R_id],R)
            if R_diff<min_diff:
                min_diff=R_diff
                best_id=R_id
        return best_id


    def DeltaR(self,R,index):
        R_anchor=self.Rgroup[index]#3*3
        #R=Rres@Ranc->Rres=R@Ranc.T
        deltaR=R@R_anchor.T
        return quaternion_from_matrix(deltaR)


    def Gen_trainset(self):
        Save_list_dir=f'{self.output_dir}/trainset'
        make_non_exists_dir(Save_list_dir)
        batch_i=-1
        trainlist_pair=[]
        for name,dataset in tqdm(self.datasets.items()):
            if name in ['wholesetname','valscenes']:continue
            if name in self.valscenes:
                print(f'val scene: {name}')
                continue
            for pair in tqdm(dataset.pair_ids):
                pc0,pc1=pair
                #if os.path.exists(f'{Save_list_dir}/{i*16)}.pth'):continue
                #feature readin
                Feats0=np.load(f'{self.output_dir}/Rotated_Features/{dataset.name}/{pc0}_feats.npz')
                Feats1=np.load(f'{self.output_dir}/Rotated_Features/{dataset.name}/{pc1}_feats.npz')
                Feats0_f=Feats0['feats']
                Feats1_f=Feats1['feats']
                Feats0_R=Feats0['Rs']
                Feats1_R=Feats1['Rs']
                AllRs=[]
                AllR_indexs=[]
                AlldeltaRs=[]
                gtR = dataset.get_transform(pc0,pc1)[0:3,0:3]
                for Ri_id in range(Feats0_R.shape[0]):
                    for Rj_id in range(Feats1_R.shape[0]):
                        R_i=Feats0_R[Ri_id]
                        R_j=Feats1_R[Rj_id]
                        R=R_j@gtR.T@R_i.T  #note: here is pc0 to pc1
                        true_idx=self.R2DR_id(R)
                        delR=self.DeltaR(R,true_idx)
                        AllRs.append(R[None,:,:])
                        AllR_indexs.append(true_idx)
                        AlldeltaRs.append(delR[None,:])
                AllRs=np.concatenate(AllRs,axis=0).reshape([self.Rnum,self.Rnum,3,3])
                AllR_indexs=np.array(AllR_indexs).reshape([self.Rnum,self.Rnum])
                AlldeltaRs=np.concatenate(AlldeltaRs,axis=0).reshape([self.Rnum,self.Rnum,4])

                #pps
                Key_pps=np.load(f'{self.output_dir}/Pairs/{dataset.name}/{pc0}-{pc1}.npy') #index in keys
                pps_all=np.arange(Key_pps.shape[0]) #index
                
                if pps_all.shape[0]<10:continue
                if pps_all.shape[0]<self.cfg.batch_size:
                    pps_all=np.repeat(pps_all,int(self.cfg.batch_size/pps_all.shape[0])+1)
                    np.random.shuffle(pps_all)
                
                for i in range(10):
                    #pair pps (choose self.cfg.batch_size):
                    np.random.shuffle(pps_all)
                    pps=Key_pps[pps_all[0:self.cfg.batch_size]]# bn*2
                    BaseIndex=np.arange(self.Rnum).astype(np.int)
                    Index_i=np.random.choice(BaseIndex, size=self.cfg.batch_size, replace=True)
                    Index_j=np.random.choice(BaseIndex, size=self.cfg.batch_size, replace=True)
                    
                    Rs=[]
                    R_indexs=[]
                    deltaR=[]
                    feats_one_batch_i=[]
                    feats_one_batch_j=[]
                    for b in range(self.cfg.batch_size):
                        Rs.append(AllRs[Index_i[b],Index_j[b]][None,:,:])
                        R_indexs.append(AllR_indexs[Index_i[b],Index_j[b]])
                        deltaR.append(AlldeltaRs[Index_i[b],Index_j[b]][None,:])
                        #feat
                        feats_one_batch_i.append(Feats0_f[Index_i[b],pps[b,0]][None,:,:])
                        feats_one_batch_j.append(Feats1_f[Index_j[b],pps[b,1]][None,:,:])
                    Rs=np.concatenate(Rs,axis=0)
                    R_indexs=np.array(R_indexs)
                    deltaR=np.concatenate(deltaR,axis=0)
                    feats_one_batch_i=np.concatenate(feats_one_batch_i,axis=0)
                    feats_one_batch_j=np.concatenate(feats_one_batch_j,axis=0)

                    item={
                        'feats0':torch.from_numpy(feats_one_batch_i.astype(np.float32)), #before enhanced rot
                        'feats1':torch.from_numpy(feats_one_batch_j.astype(np.float32)), #after enhanced rot
                        'R':torch.from_numpy(Rs.astype(np.float32)),
                        'true_idx':torch.from_numpy(R_indexs.astype(np.int)),
                        'deltaR':torch.from_numpy(deltaR.astype(np.float32))
                    }
                    batch_i+=1
                    torch.save(item,f'{Save_list_dir}/{batch_i}.pth',_use_new_zipfile_serialization=False)
                    trainlist_pair.append((dataset.name,pc0,pc1,i))
            save_pickle(trainlist_pair,f'{self.output_dir}/train_GFET.pkl')
        


    def Gen_valset(self):
        Save_list_dir=f'{self.output_dir}/valset'
        make_non_exists_dir(Save_list_dir)
        val_pc_pts=[]
        if not os.path.exists(f'{self.output_dir}/val_GFET.pkl'):
            for scene in tqdm(self.valscenes):
                dataset=self.datasets[scene]
                for pair in tqdm(dataset.pair_ids):
                    pc0,pc1=pair
                    Key_pps=np.load(f'{self.output_dir}/Pairs/{dataset.name}/{pc0}-{pc1}.npy') #index in keys
                    for k in range(Key_pps.shape[0]):
                        BaseIndex=np.arange(self.Rnum).astype(np.int)
                        Ri=np.random.choice(BaseIndex, size=1, replace=True)[0]
                        Rj=np.random.choice(BaseIndex, size=1, replace=True)[0]
                        val_pc_pts.append((dataset.name,pc0,pc1,Ri,Rj,Key_pps[k,0],Key_pps[k,1]))
            random.shuffle(val_pc_pts)
            val_pc_pts=val_pc_pts[0:3000]
            save_pickle(val_pc_pts,f'{self.output_dir}/val_GFET.pkl')
        else:
            val_pc_pts=read_pickle(f'{self.output_dir}/val_GFET.pkl')

        for i in tqdm(range(len(val_pc_pts))):
            datasetname,pc0,pc1,Ri_id,Rj_id,pt0,pt1=val_pc_pts[i]
            Feats0=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc0}_feats.npz')
            Feats1=np.load(f'{self.output_dir}/Rotated_Features/{datasetname}/{pc1}_feats.npz')
            datasetname=datasetname[(str.rfind(datasetname,'/')+1):]
            gtR = self.datasets[datasetname].get_transform(pc0,pc1)[0:3,0:3]
            R_i=Feats0['Rs'][Ri_id]
            R_j=Feats1['Rs'][Rj_id]
            R=R_j@gtR.T@R_i.T
            true_idx=self.R2DR_id(R)
            feat0=Feats0['feats'][Ri_id,pt0]
            feat1=Feats1['feats'][Rj_id,pt1]
            item={
                'feats0':torch.from_numpy(feat0.astype(np.float32)), #before enhanced rot 32*60
                'feats1':torch.from_numpy(feat1.astype(np.float32)), #after enhanced rot 32*60
                'R':torch.from_numpy(R.astype(np.float32)),
                'true_idx':torch.from_numpy(np.array([true_idx]))
            }
            torch.save(item,f'{Save_list_dir}/{i}.pth',_use_new_zipfile_serialization=False)

    def Clean_cache(self):
        os.system(f'rm -rf {self.output_dir}/Rotated_Features')

    def run(self):
        self.PCA_keys_sample()
        self.PC_random_rot_feat()
        self.Gen_trainset()
        self.Gen_valset()
        # clean cache
        self.Clean_cache()

if __name__=="__main__":
    args, nouse = get_config()
    trainset_creater=GF_ET_trainset(args)
    trainset_creater.PCA_keys_sample()
    trainset_creater.PC_random_rot_feat()
    # trainset_creater.trainset()
    trainset_creater.valset()
    
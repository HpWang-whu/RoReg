import os,sys
import time
sys.path.append('..')
import numpy as np
import open3d as o3d
from tqdm import tqdm
import utils.RR_cal as RR_cal
from utils.r_eval import compute_R_diff
from dataops.dataset import get_dataset_name
from utils.utils import transform_points, make_non_exists_dir
from test import name2extractor, name2detector, name2matcher, name2estimator

class yoho_evaluator:
    def __init__(self,cfg):
        self.cfg = cfg
        self.GF = self.cfg.GF 
        self.RD = self.cfg.RD               # bool
        self.RM = self.cfg.RM               # bool
        self.ET = self.cfg.ET               # str
        self.keynum = self.cfg.keynum
        self.max_iter = self.cfg.max_iter
        # initialization
        self.extractor = name2extractor[self.GF](self.cfg)
        self.detector = None
        self.matcher = name2matcher['matmul'](self.cfg)
        self.estimator = name2estimator[self.ET](self.cfg)
        # re-value
        if self.RD:
            self.RD = 'yoho_det'
            self.detector = name2detector['yoho_det'](self.cfg)
        else:
            self.RD = 'nodet'
        if self.RM:
            self.RM = 'yoho_mat'
            self.matcher = name2matcher['yoho_mat'](self.cfg)
        else:
            self.RM = 'matmul'
    
    def process_scene(self, dataset):
        # feature extractor
        self.extractor.run(dataset)
        # detector
        if self.detector is not None:
            self.detector.run(dataset)
        # matcher
        self.matcher.run(dataset, self.keynum)
        # estimator
        self.estimator.run(dataset, self.keynum, self.max_iter)
            
    def fmr_ir_scene(self, dataset):
        fmrs = []
        irs = []
        for pair in dataset.pair_ids:
            id0,id1 = pair
            # extract the correspondences
            corr = np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/match_{self.keynum}/{id0}-{id1}.npy')
            corr_s = np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/match_{self.keynum}/scores/{id0}-{id1}.npy')
            if self.cfg.RM:
                if self.cfg.match_n <0.999:
                    num=max(corr_s.shape[0]*self.cfg.match_n,10)
                else:
                    num = self.cfg.match_n
                sample_index=np.argsort(corr_s)[-int(num):]
                corr=corr[sample_index]
            keys0 = dataset.get_kps(id0)
            keys1 = dataset.get_kps(id1)
            keysm0 = keys0[corr[:,0]]
            keysm1 = keys1[corr[:,1]]
            # distance
            gt = dataset.get_transform(id0, id1)
            keysm1 = transform_points(keysm1, gt)
            dist = np.sqrt(np.sum(np.square(keysm0-keysm1),axis=-1))
            ir = np.mean(dist < self.cfg.tau_2)
            irs.append(ir)
            if ir > self.cfg.tau_1:
                fmrs.append(1)
            else:
                fmrs.append(0)
        fmr = np.mean(np.array(fmrs))
        ir = np.mean(np.array(irs))
        return fmr, ir
    
    def rr_scene(self, dataset):
        # RR,RRE,RTE of pointdsc
        rrs,rre,rte = [],[],[]
        for pair in dataset.pair_ids:
            id0,id1 = pair
            gt = dataset.get_transform(id0,id1)
            trans = np.load(f'{self.cfg.output_cache_fn}/{dataset.name}/match_{self.keynum}/{self.ET}/{self.max_iter}iters/{id0}-{id1}.npz')['trans']
            Rpre, tpre = trans[0:3,0:3], trans[0:3,-1]
            Rgt, tgt = gt[0:3,0:3], gt[0:3,-1]
            Rdiff = compute_R_diff(Rpre, Rgt)
            tdiff = np.sqrt(np.sum(np.square(tpre-tgt)))
            if (Rdiff<15) and (tdiff<0.3):
                rrs.append(1)
                # Following pointdsc, rre and rte are calculated only on the successful cases.
                rre.append(Rdiff)
                rte.append(tdiff)
            else:
                rrs.append(0)
        return np.mean(np.array(rrs)), np.mean(np.array(rre)), np.mean(np.array(rte))
                
    def run(self):
        testset = self.cfg.testset
        datasets = get_dataset_name(testset,self.cfg.origin_data_dir)
        # process
        for name, dataset in datasets.items():
            if type(dataset) is str: continue
            self.process_scene(dataset)
        # fmr and is
        fmrs, irs = [], []
        for name, dataset in datasets.items():
            if type(dataset) is str: continue
            fmr, ir = self.fmr_ir_scene(dataset)
            fmrs.append(fmr)
            irs.append(ir)
        fmr = np.mean(np.array(fmrs))        
        ir = np.mean(np.array(irs))        
        # rr pointdsc
        rr_dsc, rre_dsc, rte_dsc = [],[],[]
        for name, dataset in datasets.items():
            if type(dataset) is str: continue
            rr,rre,rte = self.rr_scene(dataset)
            rr_dsc.append(rr)
            rre_dsc.append(rre)
            rte_dsc.append(rte)
        rr_dsc = np.mean(np.array(rr_dsc))   
        rre_dsc = np.mean(np.array(rre_dsc))   
        rte_dsc = np.mean(np.array(rte_dsc))               
        # rr
        if datasets['wholesetname'] == 'demo': rr_predator = 1.0
        else: rr_predator,_,_=RR_cal.benchmark(self.cfg,datasets,self.keynum,self.max_iter,yoho_sign=self.ET)
        # output
        datasetname = datasets['wholesetname']
        msg=f'{datasetname}-{self.GF}-{self.RD}-{self.RM}-{self.ET}-{self.keynum}keys-{self.max_iter}iters\n'
        msg+=f'feature matching recall          : {fmr:.5f}\n' \
             f'inlier ratio                     : {ir:.5f}\n' \
             f'registration recall(predator)    : {rr_predator:.5f}\n' \
             f'rotation error(pointdsc)         : {rre_dsc:.5f}\n' \
             f'translation error(pointdsc)      : {rte_dsc:.5f}\n' \
             f'registration recall(pointdsc)    : {rr_dsc:.5f}'

        with open(f'{self.cfg.base_dir}/results.log','a') as f:
            f.write(msg+'\n')
        print(msg)

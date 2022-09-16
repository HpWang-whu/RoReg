import os
import numpy as np
from dataops.dataset import get_dataset_name
from utils.utils_o3d import draw_registration_result
os.environ['MKL_THREADING_LAYER'] = 'GNU' 

# execuate fcgf backbone
os.system(f'python testset.py --dataset demo')

# conduct yoho
os.system('python Test.py --testset demo --RD --RM --ET yohoo --keynum 250 --max_iter 1000')

# draw results
dataset = get_dataset_name('demo', '../data/origin_data')['kitchen']
yohoo_result=np.load(f'./data/YOHO_FCGF/Testset/demo/kitchen/match_250/yohoo/1000iters/0-1.npz')
yohoo_trans=yohoo_result['trans'].astype(np.float64)
    
#visual
target=dataset.get_pc_o3d('0')
source=dataset.get_pc_o3d('1')
#origin
draw_registration_result(source,target,np.eye(4))
#predict
draw_registration_result(source,target,yohoo_trans)

# Import python dependencies
import argparse


base_dir='./data'
backbone='FCGF'
arg_lists = []
parser = argparse.ArgumentParser()

def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg

Dirs=add_argument_group('Dirs')
Dataset_Args=add_argument_group('Dataset')
Train_Args=add_argument_group("Training_Args")
Val_Args=add_argument_group("Validation_Args")
Network_Args=add_argument_group("Network_Args")
Test_Args=add_argument_group("Test_Args")

############################################# Base ###################################################
#Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,
                        help="base dir containing the whole project")
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}/origin_data",
                        help="the dir containing whole datas")
Dirs.add_argument("--backbone",type=str,default=backbone,
                        help="name of backbone")
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/YOHO_{backbone}/Testset",
                        help="eval cache dir")
Dirs.add_argument("--model_fn",type=str,default=f"./checkpoints/{backbone}",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./utils/group_related",
                        help='SO3 related files path')

############################################# Test pipeline ###################################################
#Input
Test_Args.add_argument('--GF',default='yoho_des',type=str)
Test_Args.add_argument('--RD',action='store_true')
Test_Args.add_argument('--RM',action='store_true')
Test_Args.add_argument('--ET',default='yohoc',type=str,help = 'ransac/yohoc/yohoo')

Test_Args.add_argument('--testset',default='3dmatch',type=str,help='testset name')
Test_Args.add_argument('--keynum',default=5000,type=int,help='number of keypoints')
Test_Args.add_argument('--max_iter',default=1000,type=int,help='ransac iterations')
Test_Args.add_argument('--ransac_ird',default=0.1,type=float,help='inliner threshold of ransac')
Test_Args.add_argument('--tau_1',default=0.05,type=float,help='tau 1 for FMR, 5%')
Test_Args.add_argument('--tau_2',default=0.1,type=float,help='tau 2 for FMR, 0.1m')
Test_Args.add_argument('--tau_3',default=0.2,type=float,help='tau 3 for RR, 0.2m')
Test_Args.add_argument('--match_n',default=0.5,type=float,help='use how many correspondences predicted for transformation estimation, 0.99 to use all, if n>=1, use top-n')

Test_Args.add_argument('--bs_GF',default=1250,type=int,help='test batch size for group feature extraction')
Test_Args.add_argument('--bs_ET',default=1000,type=int,help='test batch size for local transformation estimation')


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



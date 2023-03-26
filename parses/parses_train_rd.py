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

############################################# Base ###################################################
# Dirs
Dirs.add_argument('--base_dir',type=str,default=base_dir,
                        help="base dir containing the whole project")
Dirs.add_argument("--origin_data_dir",type=str,default=f"{base_dir}/origin_data",
                        help="the dir containing whole datas")
Dirs.add_argument("--backbone",type=str,default=backbone,
                        help="name of backbone")
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/YOHO_{backbone}/Trainset",
                        help="eval cache dir")
Dirs.add_argument("--model_fn",type=str,default=f"./checkpoints/{backbone}",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./utils/group_related",
                        help='SO3 related files path')

###################################### Trainset generation for RD ##############################################
# generate fcgf input  for RD
Dataset_Args.add_argument("--part",type=str,default="RD",
                        help="short name(generating trainset for which component)")
Dataset_Args.add_argument("--backbone_w",type=str,default=f"./checkpoints/{backbone}/backbone/best_val_checkpoint.pth",
                        help="the file of the backbone checkpoint")
Dataset_Args.add_argument("--trainset",type=str,default="3dm_train_rot",
                        help="train dataset name")
Dataset_Args.add_argument("--voxelsize",type=float,default=0.025,
                        help="voxelsize for fcgf backbone")
Dataset_Args.add_argument("--keynum",type=int,default=5000,
                        help="YOHO points in scan")
Dataset_Args.add_argument("--bs_GF",type=int,default=900,
                        help="batch size for testset generation of partI")
Dataset_Args.add_argument("--bs_ET",type=int,default=1000,
                        help="batch size for testset generation of partII")

####################################### Training for RD ##########################################################
#general
Train_Args.add_argument("--loss_type",type=str,default="loss_rd",
                        help="loss type")
Train_Args.add_argument("--val_type",type=str,default="val_rd",
                        help="val_type")
Train_Args.add_argument("--trainset_type",type=str,default="dataset_rd",
                        help="trainset_type")

# hyperparameters
Train_Args.add_argument("--batch_size",type=int,default=128,
                        help="Training batch size")
Train_Args.add_argument("--batch_size_val",type=int,default=128,
                        help="Training batch size")
Train_Args.add_argument("--worker_num",type=int,default=16,
                        help="the threads used for dataloader")
Train_Args.add_argument("--epochs",type=int,default=2,
                        help="num of epoches")
Train_Args.add_argument("--multi_gpus",type=bool,default=False,
                        help="whether use the mutli gpus")
Train_Args.add_argument("--lr_init",type=float,default=0.001,
                        help="The initial learning rate")
Train_Args.add_argument("--lr_decay_rate",type=float,default=0.8,
                        help="the decay rate of the learning rate per epoch")
Train_Args.add_argument("--lr_decay_step",type=float,default=0.8,
                        help="the decay step of the learning rate (how many epoches)")

#log
Train_Args.add_argument("--train_log_step",type=int,default=500,
                        help="logger internal")
Val_Args.add_argument("--val_interval",type=int,default=3000,
                        help="the interval to validation")
Val_Args.add_argument("--save_interval",type=int,default=1000,
                        help="the interval to save the model")

# datalist 
Dataset_Args.add_argument("--trainlist",type=str,default=f'{base_dir}/YOHO_{backbone}/Trainset/3dm_train_rot/train_RD.pkl',
                        help="validation tuples (station,pc0,pc1,R_i,R_j,pt0,pt1)")
Dataset_Args.add_argument("--vallist",type=str,default=f'{base_dir}/YOHO_{backbone}/Trainset/3dm_train_rot/val_RD.pkl',
                        help="validation tuples (station,pc0,pc1,R_i,R_j,pt0,pt1)")


def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



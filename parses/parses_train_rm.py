# Import python dependencies
# config for training rotation coherence matcher
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
Dirs.add_argument("--output_cache_fn",type=str,default=f"{base_dir}/YOHO_{backbone}/Trainset",
                        help="eval cache dir for 3dm_train_rot+YOHO_Des")
Dirs.add_argument("--model_fn",type=str,default=f"./checkpoints/{backbone}",
                        help='well trained model path')
Dirs.add_argument('--SO3_related_files',type=str,default=f"./utils/group_related",
                        help='SO3 related files path')

###################################### Trainset generation for RD and RM ##############################################
Dataset_Args.add_argument("--part",type=str,default="RM",
                        help="short name(generating trainset for which component)")
Dataset_Args.add_argument("--trainset",type=str,default="3dm_train_rot",
                        help="train dataset name")
Dataset_Args.add_argument("--min_ps",type=int,default=256,
                        help="min sample points per scan in RM")
Dataset_Args.add_argument("--max_ps",type=int,default=1538,
                        help="max sample points per scan in RM")
Dataset_Args.add_argument("--pps_thre",type=float,default=0.2,
                        help="threshold for pp matching")

############################################# Train ###################################################
#Train Args
#general
Train_Args.add_argument("--loss_type",type=str,default="loss_rm",
                        help="loss type")
Train_Args.add_argument("--val_type",type=str,default="val_rm",
                        help="val_type")
Train_Args.add_argument("--trainset_type",type=str,default="dataset_rm",
                        help="trainset_type")

# hyperparameters
Train_Args.add_argument("--batch_size",type=int,default=1,
                        help="Training batch size")
Train_Args.add_argument("--batch_size_val",type=int,default=1,
                        help="Training batch size")
Train_Args.add_argument("--worker_num",type=int,default=16,
                        help="the threads used for dataloader")
Train_Args.add_argument("--epochs",type=int,default=3,
                        help="num of epoches")
Train_Args.add_argument("--multi_gpus",type=bool,default=False,
                        help="whether use the mutli gpus")
Train_Args.add_argument("--lr_init",type=float,default=0.001,
                        help="The initial learning rate")
Train_Args.add_argument("--lr_decay_rate",type=float,default=0.8,
                        help="the decay rate of the learning rate per epoch")
Train_Args.add_argument("--lr_decay_step",type=float,default=1,
                        help="the decay step of the learning rate (how many epoches)")

#log
Train_Args.add_argument("--train_log_step",type=int,default=1,
                        help="logger internal")
Val_Args.add_argument("--val_interval",type=int,default=10,
                        help="the interval to validation")
Val_Args.add_argument("--save_interval",type=int,default=2,
                        help="the interval to save the model")

# datalist 
Dataset_Args.add_argument("--trainlist",type=str,default=f'{base_dir}/YOHO_{backbone}/Trainset/RM/train_RM.pkl',
                        help="validation tuples (station,pc0,pc1,R_i,R_j,pt0,pt1)")
Dataset_Args.add_argument("--vallist",type=str,default=f'{base_dir}/YOHO_{backbone}/Trainset/RM/val_RM.pkl',
                        help="validation tuples (station,pc0,pc1,R_i,R_j,pt0,pt1)")

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed

def print_usage():
    parser.print_usage()



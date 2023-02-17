from train.loss import *
from train.val import *
from train.trainset import *

name2trainset={
    'trainset_gf':GF_ET_trainset,
    'trainset_rd':RD_trainset,
    'trainset_rm':RM_trainset
}
from test.extractor import *
from test.detector import *
from test.matcher import *
from test.estimator import *

name2extractor={
    'yoho_des':yoho_des
}

name2detector={
    'yoho_det':yoho_det
}

name2matcher={
    'matmul':mutual,
    'yoho_mat':yoho_mat
}

name2estimator={
    'yohoc':yohoc,
    'yohoo':yohoo
}

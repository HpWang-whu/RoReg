from network.group_feat import GF_train, GF_test
from network.rot_detect import detector_eqv, detector_eqv_test
from network.rot_coh_match import Match_ot
from network.eqv_trans import ET_train, ET_test


name2network={
    # group feature extractor
    'GF_train':GF_train,
    'GF_test':GF_test,
    # rotation guided detector
    'RD_train':detector_eqv,
    'RD_test':detector_eqv_test,
    # rotation coherence matcher
    'RM_train':Match_ot,
    'RM_test':Match_ot,
    # transformation estimation
    'ET_train':ET_train,
    'ET_test':ET_test
}
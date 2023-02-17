import parses.parses_gf as parses_gf
import parses.parses_rd as parses_rd
import parses.parses_rm as parses_rm
from train import name2trainset

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--component',
    default='GF',
    type=str,
    help='GF/RD/RM for indicating which kind of trainset are generated')
args = parser.parse_args()

if args.component == 'GF':
    cfg,_ = parses_gf.get_config()
    generator = name2trainset['trainset_gf'](cfg)
    generator.run()
    
elif args.component == 'RD':
    cfg,_ = parses_rd.get_config()
    generator = name2trainset['trainset_rd'](cfg)
    generator.run()
    
elif args.component == 'RM':
    cfg,_ = parses_rm.get_config()
    generator = name2trainset['trainset_rm'](cfg)
    generator.run()

else:
    print('wrong sign, choose one from GF/RD/RM')

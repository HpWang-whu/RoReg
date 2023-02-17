import parses.parses_train_gf as parses_gf
import parses.parses_train_rd as parses_rd
import parses.parses_train_rm as parses_rm
import parses.parses_train_et as parses_et
from train.trainer import name2trainer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--component',
    default='GF',
    type=str,
    help='GF/RD/RM/ET for indicating which kind of trainset are generated')
args = parser.parse_args()

if args.component == 'GF':
    cfg,_ = parses_gf.get_config()
    generator = name2trainer['trainer_gf'](cfg)
    generator.run()
    
elif args.component == 'RD':
    cfg,_ = parses_rd.get_config()
    generator = name2trainer['trainer_rd'](cfg)
    generator.run()
    
elif args.component == 'RM':
    cfg,_ = parses_rm.get_config()
    generator = name2trainer['trainer_rm'](cfg)
    generator.run()
    
elif args.component == 'ET':
    cfg,_ = parses_et.get_config()
    generator = name2trainer['trainer_et'](cfg)
    generator.run()

else:
    print('wrong sign, choose one from GF/RD/RM/ET')
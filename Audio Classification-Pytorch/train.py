import argparse
import functools

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',      'configuration file')
add_arg("local_rank",       int,    0,                             'Parameters needed for multi-card training')
add_arg("use_gpu",          bool,   False,                          'Whether or not to use GPU training')
add_arg('augment_conf_path',str,    'configs/augmentation.json',   'Configuration file for data enhancement in json format')
add_arg('save_model_path',  str,    'models/',                  'Path where the model is saved')
add_arg('resume_model',     str,    None,                       'Resume training, when None then no pre-trained model is used')
add_arg('pretrained_model', str,    None,                       'Path of the pre-trained model, when None then the pre-trained model is not used')
args = parser.parse_args()
print_arguments(args=args)

# Get the trainer
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

trainer.train(save_model_path=args.save_model_path,
              resume_model=args.resume_model,
              pretrained_model=args.pretrained_model,
              augment_conf_path=args.augment_conf_path)

import argparse
import functools
import time

from macls.trainer import MAClsTrainer
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,   'configs/ecapa_tdnn.yml',    "configuration file")
add_arg("use_gpu",          bool,  False,                        "Whether or not to use GPU evaluation models")
add_arg('save_matrix_path', str,   'output/images/',            "Save the path to the mixing matrix")
add_arg('resume_model',     str,   'models/EcapaTdnn_MFCC/best_model/',  "Path of the model")
args = parser.parse_args()
print_arguments(args=args)

# Get the trainer
trainer = MAClsTrainer(configs=args.configs, use_gpu=args.use_gpu)

# Commencement of assessment
start = time.time()
loss, accuracy = trainer.evaluate(resume_model=args.resume_model,
                                  save_matrix_path=args.save_matrix_path)
end = time.time()
print('Assessment of consumption time：{}s，loss：{:.5f}，accuracy：{:.5f}'.format(int(end - start), loss, accuracy))

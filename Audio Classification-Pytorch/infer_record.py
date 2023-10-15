import argparse
import functools

from macls.predict import MAClsPredictor
from macls.utils.record import RecordAudio
from macls.utils.utils import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    'configs/ecapa_tdnn.yml',   'configuration file')
add_arg('use_gpu',          bool,   True,                       'Whether to use GPU prediction')
add_arg('record_seconds',   int,    3,                          'Length of recording')
add_arg('model_path',       str,    'models/EcapaTdnn_MelSpectrogram/best_model/', 'Path to the exported predictive model file')
args = parser.parse_args()
print_arguments(args=args)

# Get Identifier
predictor = MAClsPredictor(configs=args.configs,
                           model_path=args.model_path,
                           use_gpu=args.use_gpu)

record_audio = RecordAudio()

if __name__ == '__main__':
    try:
        while True:
            # Load data
            input(f"Press the Enter key to switch on the recording, recording{args.record_seconds}seconds：")
            audio_data = record_audio.record(record_seconds=args.record_seconds)
            # Access to forecast results
            label, s = predictor.predict(audio_data, sample_rate=record_audio.sample_rate)
            print(f'The predicted labels are：{label}，score：{s}')
    except Exception as e:
        print(e)

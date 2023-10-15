import argparse
import functools
import moviepy.editor as mpe
from moviepy.editor import AudioFileClip
from macls.predict import MAClsPredictor
from macls.utils.utils import add_arguments, print_arguments
from pydub import AudioSegment
import os
import os
FPS = 16000  # stated frame rate
NBYTES = 16  # declarator
FFMPEG_PARAMS = ["-ac", "1"]  # mono setting
def mp4_to_wav(input_file, output_folder):
    #Convert input files to Pydub format
    sound = AudioSegment.from_file(input_file, format="mp4")
    #Save the list of .wav files in the output folder as a variable
    wav_files = [f for f in os.listdir(output_folder) if f.endswith('.wav')]
    #Set the number of output .wav files equal to the number of .wav files in the corresponding folder
    file_number = len(wav_files) + 1
    #Set the name of the output file to "%d.wav" % file_number
    output_file = os.path.join(output_folder, '%d.wav' % file_number)
    #Export Pydub format files to .wav format
    sound.export(output_file, format="wav")
    #Output successful conversion message
    print('Conversion successful')

# mp4_to_wav(c,'./1')
# audio = AudioFileClip('./V_DRONE_111.mp4')
# audio.write_audiofile('./3333.wav')
# print('1',audio)
# aa
# import subprocess

# command ="ffmpeg -i ./V_HELICOPTER_056.mp4 -ab 160k -ac 2 -ar 44100 -vn ./V_HELICOPTER_056.wav"
# command ="ffmpeg -i ./V_HELICOPTER_056.mp4 ./V_HELICOPTER_056.wav"

# subprocess.call(command, shell=True)

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('configs',          str,    './configs/ecapa_tdnn.yml',   'configuration file')
add_arg('use_gpu',          bool,   False,                       'Whether to use GPU prediction')
add_arg('audio_path',       str,    './V_DRONE_111.mov', 'audio path')
add_arg('model_path',       str,    './models/EcapaTdnn_MFCC/best_model/', 'Path to the exported predictive model file')
args = parser.parse_args()
print_arguments(args=args)

# Get Identifier
predictor = MAClsPredictor(configs=args.configs,                           model_path=args.model_path,                           use_gpu=args.use_gpu)

with open('./dataset/test_list.txt','r') as g:
    l = g.readlines()

for ll in l:
    path = ll.split('\t')[0]
    print(path)
    #
    label, score = predictor.predict(audio_data=path)
    #
    with open('./result.txt', 'a') as g:
        g.write(f'sound frequency：{path} The predicted results are labelled as：{label}，score：{score}\n')
    print(f'sound frequency：{path} The predicted results are labelled as：{label}，score：{score}')

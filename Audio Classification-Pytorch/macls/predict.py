import os
from io import BufferedReader
from typing import List

import numpy as np
import torch
import yaml

from macls import SUPPORT_MODEL
from macls.data_utils.audio import AudioSegment
from macls.data_utils.featurizer import AudioFeaturizer
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE
from macls.models.tdnn import TDNN
from macls.utils.logger import setup_logger
from macls.utils.utils import dict_to_object, print_arguments

logger = setup_logger(__name__)


class MAClsPredictor:
    def __init__(self,
                 configs,
                 model_path='models/EcapaTdnn_MFCC/best_model/',
                 use_gpu=True):
        """
        Sound Classification Prediction Tool
        :param configs: configuration parameters
        :param model_path: path to the exported prediction model folder
        :param use_gpu: if or not to use GPU prediction
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU unavailable'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        # Read configuration file
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'The model is not available:{self.configs.use_model}'
        # Getting the characteriser
        self._audio_featurizer = AudioFeaturizer(feature_conf=self.configs.feature_conf, **self.configs.preprocess_conf)
        self._audio_featurizer.to(self.device)
        # Getting the model
        if self.configs.use_model == 'EcapaTdnn':
            self.predictor = EcapaTdnn(input_size=self._audio_featurizer.feature_dim,
                                       num_class=self.configs.dataset_conf.num_class,
                                       **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN6':
            self.predictor = PANNS_CNN6(input_size=self._audio_featurizer.feature_dim,
                                        num_class=self.configs.dataset_conf.num_class,
                                        **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN10':
            self.predictor = PANNS_CNN10(input_size=self._audio_featurizer.feature_dim,
                                         num_class=self.configs.dataset_conf.num_class,
                                         **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN14':
            self.predictor = PANNS_CNN14(input_size=self._audio_featurizer.feature_dim,
                                         num_class=self.configs.dataset_conf.num_class,
                                         **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Net':
            self.predictor = Res2Net(input_size=self._audio_featurizer.feature_dim,
                                     num_class=self.configs.dataset_conf.num_class,
                                     **self.configs.model_conf)
        elif self.configs.use_model == 'ResNetSE':
            self.predictor = ResNetSE(input_size=self._audio_featurizer.feature_dim,
                                      num_class=self.configs.dataset_conf.num_class,
                                      **self.configs.model_conf)
        elif self.configs.use_model == 'TDNN':
            self.predictor = TDNN(input_size=self._audio_featurizer.feature_dim,
                                  num_class=self.configs.dataset_conf.num_class,
                                  **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} The model does not exist!')
        self.predictor.to(self.device)
        # Loading Models
        if os.path.isdir(model_path):
            model_path = os.path.join(model_path, 'model.pt')
        assert os.path.exists(model_path), f"{model_path} The model does not exist!"
        if torch.cuda.is_available() and use_gpu:
            model_state_dict = torch.load(model_path)
        else:
            model_state_dict = torch.load(model_path, map_location='cpu')
        self.predictor.load_state_dict(model_state_dict)
        print(f"Successfully loaded model parameters:{model_path}")
        self.predictor.eval()
        # Get Category Tags
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]

    def _load_audio(self, audio_data, sample_rate=16000):
        """Load audio
        :param audio_data: the data to be recognised, support file path, file object, byte, numpy, if byte, must be a full byte file.
        :param sample_rate: if you pass in numpy data, you need to specify the sample rate.
        :return: the result of the recognised text and the decoded score.
        """
        # Load audio files and pre-process them
        if isinstance(audio_data, str):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, BufferedReader):
            audio_segment = AudioSegment.from_file(audio_data)
        elif isinstance(audio_data, np.ndarray):
            audio_segment = AudioSegment.from_ndarray(audio_data, sample_rate)
        elif isinstance(audio_data, bytes):
            audio_segment = AudioSegment.from_bytes(audio_data)
        else:
            raise Exception(f'This data type is not supported, the current data type is：{type(audio_data)}')
        # resample
        if audio_segment.sample_rate != self.configs.dataset_conf.sample_rate:
            audio_segment.resample(self.configs.dataset_conf.sample_rate)
        # decibel normalization
        if self.configs.dataset_conf.use_dB_normalization:
            audio_segment.normalize(target_db=self.configs.dataset_conf.target_dB)
        return audio_segment

    # Predicting the characteristics of an audio
    def predict(self,
                audio_data,
                sample_rate=16000):
        """Predict an audio

        :param audio_data: the data to be recognised, support file path, file object, byte, numpy, if it is byte, it must be a complete byte file with format.
        :param sample_rate: the sample rate to be specified if numpy data is passed in.
        :return: result label and corresponding score
        """
        # Load audio files and pre-process them
        input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
        assert input_data.duration >= self.configs.dataset_conf.min_duration, \
            f'Audio is too short, the minimum should be{self.configs.dataset_conf.min_duration}s，The current audio is{input_data.duration}s'
        input_data = torch.tensor(input_data.samples, dtype=torch.float32, device=self.device).unsqueeze(0)
        input_len_ratio = torch.tensor([1], dtype=torch.float32, device=self.device)
        audio_feature, _ = self._audio_featurizer(input_data, input_len_ratio)
        # Implementation projections
        output = self.predictor(audio_feature)
        result = torch.nn.functional.softmax(output, dim=-1)[0]
        result = result.data.cpu().numpy()
        # Maximum probability of label
        lab = np.argsort(result)[-1]
        score = result[lab]
        return self.class_labels[lab], round(float(score), 5)

    def predict_batch(self, audios_data: List, sample_rate=16000):
        """Predict the characteristics of a batch of audio

        :param audios_data: the data to be recognised, supports file path, file object, byte, numpy, if it is byte, it must be a complete byte file with format.
        :param sample_rate: If you are passing numpy data, you need to specify the sample rate.
        :return: result label and corresponding score
        """
        audios_data1 = []
        for audio_data in audios_data:
            # Load audio files and pre-process them
            input_data = self._load_audio(audio_data=audio_data, sample_rate=sample_rate)
            audios_data1.append(input_data.samples)
        # Find the longest audio length
        batch = sorted(audios_data1, key=lambda a: a.shape[0], reverse=True)
        max_audio_length = batch[0].shape[0]
        batch_size = len(batch)
        # Create 0 tensor with maximum length
        inputs = np.zeros((batch_size, max_audio_length), dtype='float32')
        input_lens_ratio = []
        for x in range(batch_size):
            tensor = audios_data1[x]
            seq_length = tensor.shape[0]
            # Inserting data into the all-0 tensor implements the padding
            inputs[x, :seq_length] = tensor[:]
            input_lens_ratio.append(seq_length / max_audio_length)
        input_lens_ratio = torch.tensor(input_lens_ratio, dtype=torch.float32, device=self.device)
        inputs = torch.tensor(inputs, dtype=torch.float32, device=self.device)
        audio_feature, _ = self._audio_featurizer(inputs, input_lens_ratio)
        # Implementation projections
        output = self.predictor(audio_feature)
        results = torch.nn.functional.softmax(output, dim=-1)
        results = results.data.cpu().numpy()
        labels, scores = [], []
        for result in results:
            lab = np.argsort(result)[-1]
            score = result[lab]
            labels.append(self.class_labels[lab])
            scores.append(round(float(score), 5))
        return labels, scores

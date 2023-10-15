import io
import json
import os
import platform
import shutil
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
import yaml
from sklearn.metrics import confusion_matrix
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm
from visualdl import LogWriter

from macls import SUPPORT_MODEL, __version__
from macls.data_utils.collate_fn import collate_fn
from macls.data_utils.featurizer import AudioFeaturizer
from macls.data_utils.reader import CustomDataset
from macls.models.ecapa_tdnn import EcapaTdnn
from macls.models.panns import PANNS_CNN6, PANNS_CNN10, PANNS_CNN14
from macls.models.res2net import Res2Net
from macls.models.resnet_se import ResNetSE
from macls.models.tdnn import TDNN
from macls.utils.logger import setup_logger
from macls.utils.utils import dict_to_object, plot_confusion_matrix, print_arguments

logger = setup_logger(__name__)


class MAClsTrainer(object):
    def __init__(self, configs, use_gpu=True):
        """ macls integration tool class

        :param configs: configuration dictionary
        :param use_gpu: Whether to use GPUs to train models.
        """
        if use_gpu:
            assert (torch.cuda.is_available()), 'GPU unavailable'
            self.device = torch.device("cuda")
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            self.device = torch.device("cpu")
        self.use_gpu = use_gpu
        # Read configuration file
        if isinstance(configs, str):
            with open(configs, 'r', encoding='utf-8') as f:
                configs = yaml.load(f.read(), Loader=yaml.FullLoader)
            print_arguments(configs=configs)
        self.configs = dict_to_object(configs)
        assert self.configs.use_model in SUPPORT_MODEL, f'The model is not available:{self.configs.use_model}'
        self.model = None
        self.test_loader = None
        # Get Category Tags
        with open(self.configs.dataset_conf.label_list_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.class_labels = [l.replace('\n', '') for l in lines]
        if platform.system().lower() == 'windows':
            self.configs.dataset_conf.num_workers = 0
            logger.warning('Windows systems do not support multi-threaded reading of data and have been automatically disabled!')
        # Getting the characteriser
        self.audio_featurizer = AudioFeaturizer(feature_conf=self.configs.feature_conf, **self.configs.preprocess_conf)

    def __setup_dataloader(self, augment_conf_path=None, is_train=False):
        # Getting training data
        if augment_conf_path is not None and os.path.exists(augment_conf_path) and is_train:
            augmentation_config = io.open(augment_conf_path, mode='r', encoding='utf8').read()
        else:
            if augment_conf_path is not None and not os.path.exists(augment_conf_path):
                logger.info('Data enhancement profile {} does not exist'.format(augment_conf_path))
            augmentation_config = '{}'
        if is_train:
            self.train_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.train_list,
                                               do_vad=self.configs.dataset_conf.do_vad,
                                               max_duration=self.configs.dataset_conf.max_duration,
                                               min_duration=self.configs.dataset_conf.min_duration,
                                               augmentation_config=augmentation_config,
                                               sample_rate=self.configs.dataset_conf.sample_rate,
                                               use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                               target_dB=self.configs.dataset_conf.target_dB,
                                               mode='train')
            train_sampler = None
            if torch.cuda.device_count() > 1:
                # Setting up support for multi-card training
                train_sampler = DistributedSampler(dataset=self.train_dataset)
            self.train_loader = DataLoader(dataset=self.train_dataset,
                                           collate_fn=collate_fn,
                                           shuffle=(train_sampler is None),
                                           batch_size=self.configs.dataset_conf.batch_size,
                                           sampler=train_sampler,
                                           num_workers=self.configs.dataset_conf.num_workers)
        # Getting Test Data
        self.test_dataset = CustomDataset(data_list_path=self.configs.dataset_conf.test_list,
                                          do_vad=self.configs.dataset_conf.do_vad,
                                          max_duration=self.configs.dataset_conf.max_duration,
                                          min_duration=self.configs.dataset_conf.min_duration,
                                          sample_rate=self.configs.dataset_conf.sample_rate,
                                          use_dB_normalization=self.configs.dataset_conf.use_dB_normalization,
                                          target_dB=self.configs.dataset_conf.target_dB,
                                          mode='eval')
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=self.configs.dataset_conf.batch_size,
                                      collate_fn=collate_fn,
                                      num_workers=self.configs.dataset_conf.num_workers)

    def __setup_model(self, input_size, is_train=False):
        # Getting the model
        if self.configs.use_model == 'EcapaTdnn':
            self.model = EcapaTdnn(input_size=input_size,
                                   num_class=self.configs.dataset_conf.num_class,
                                   **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN6':
            self.model = PANNS_CNN6(input_size=input_size,
                                    num_class=self.configs.dataset_conf.num_class,
                                    **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN10':
            self.model = PANNS_CNN10(input_size=input_size,
                                     num_class=self.configs.dataset_conf.num_class,
                                     **self.configs.model_conf)
        elif self.configs.use_model == 'PANNS_CNN14':
            self.model = PANNS_CNN14(input_size=input_size,
                                     num_class=self.configs.dataset_conf.num_class,
                                     **self.configs.model_conf)
        elif self.configs.use_model == 'Res2Net':
            self.model = Res2Net(input_size=input_size,
                                 num_class=self.configs.dataset_conf.num_class,
                                 **self.configs.model_conf)
        elif self.configs.use_model == 'ResNetSE':
            self.model = ResNetSE(input_size=input_size,
                                  num_class=self.configs.dataset_conf.num_class,
                                  **self.configs.model_conf)
        elif self.configs.use_model == 'TDNN':
            self.model = TDNN(input_size=input_size,
                              num_class=self.configs.dataset_conf.num_class,
                              **self.configs.model_conf)
        else:
            raise Exception(f'{self.configs.use_model} The model does not exist!')
        self.model.to(self.device)
        self.audio_featurizer.to(self.device)
        summary(self.model, input_size=(1, 98, self.audio_featurizer.feature_dim))
        # print(self.model)
        # Get the loss function
        self.loss = torch.nn.CrossEntropyLoss()
        if is_train:
            # Getting Optimisation Methods
            optimizer = self.configs.optimizer_conf.optimizer
            if optimizer == 'Adam':
                self.optimizer = torch.optim.Adam(params=self.model.parameters(),
                                                  lr=float(self.configs.optimizer_conf.learning_rate),
                                                  weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'AdamW':
                self.optimizer = torch.optim.AdamW(params=self.model.parameters(),
                                                   lr=float(self.configs.optimizer_conf.learning_rate),
                                                   weight_decay=float(self.configs.optimizer_conf.weight_decay))
            elif optimizer == 'SGD':
                self.optimizer = torch.optim.SGD(params=self.model.parameters(),
                                                 momentum=self.configs.optimizer_conf.momentum,
                                                 lr=float(self.configs.optimizer_conf.learning_rate),
                                                 weight_decay=float(self.configs.optimizer_conf.weight_decay))
            else:
                raise Exception(f'Optimisation methods are not supported:{optimizer}')
            # Learning rate decay function
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=int(self.configs.train_conf.max_epoch * 1.2))

    def __load_pretrained(self, pretrained_model):
        # Loading pre-trained models
        if pretrained_model is not None:
            if os.path.isdir(pretrained_model):
                pretrained_model = os.path.join(pretrained_model, 'model.pt')
            assert os.path.exists(pretrained_model), f"{pretrained_model} The model does not exist!"
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                model_dict = self.model.module.state_dict()
            else:
                model_dict = self.model.state_dict()
            model_state_dict = torch.load(pretrained_model)
            # Filtering non-existent parameters
            for name, weight in model_dict.items():
                if name in model_state_dict.keys():
                    if list(weight.shape) != list(model_state_dict[name].shape):
                        logger.warning('{} not used, shape {} unmatched with {} in model.'.
                                       format(name, list(model_state_dict[name].shape), list(weight.shape)))
                        model_state_dict.pop(name, None)
                else:
                    logger.warning('Lack weight: {}'.format(name))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(model_state_dict, strict=False)
            else:
                self.model.load_state_dict(model_state_dict, strict=False)
            logger.info('Successfully loaded pre-trained model:{}'.format(pretrained_model))

    def __load_checkpoint(self, save_model_path, resume_model):
        last_epoch = -1
        best_acc = 0
        last_model_dir = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'last_model')
        if resume_model is not None or (os.path.exists(os.path.join(last_model_dir, 'model.pt'))
                                        and os.path.exists(os.path.join(last_model_dir, 'optimizer.pt'))):
            # Automatic acquisition of the latest saved models
            if resume_model is None: resume_model = last_model_dir
            assert os.path.exists(os.path.join(resume_model, 'model.pt')), "The model parameter file does not exist!"
            assert os.path.exists(os.path.join(resume_model, 'optimizer.pt')), "Optimisation method parameter file does not exist!"
            state_dict = torch.load(os.path.join(resume_model, 'model.pt'))
            if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
                self.model.module.load_state_dict(state_dict)
            else:
                self.model.load_state_dict(state_dict)
            self.optimizer.load_state_dict(torch.load(os.path.join(resume_model, 'optimizer.pt')))
            with open(os.path.join(resume_model, 'model.state'), 'r', encoding='utf-8') as f:
                json_data = json.load(f)
                last_epoch = json_data['last_epoch'] - 1
                best_acc = json_data['accuracy']
            logger.info('Successful recovery of model parameters and optimisation method parameters:{}'.format(resume_model))
        return last_epoch, best_acc

    # Save the model
    def __save_checkpoint(self, save_model_path, epoch_id, best_acc=0., best_model=False):
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.model.module.state_dict()
        else:
            state_dict = self.model.state_dict()
        if best_model:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'best_model')
        else:
            model_path = os.path.join(save_model_path,
                                      f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                      'epoch_{}'.format(epoch_id))
        os.makedirs(model_path, exist_ok=True)
        torch.save(self.optimizer.state_dict(), os.path.join(model_path, 'optimizer.pt'))
        torch.save(state_dict, os.path.join(model_path, 'model.pt'))
        with open(os.path.join(model_path, 'model.state'), 'w', encoding='utf-8') as f:
            data = {"last_epoch": epoch_id, "accuracy": best_acc, "version": __version__}
            f.write(json.dumps(data))
        if not best_model:
            last_model_path = os.path.join(save_model_path,
                                           f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                           'last_model')
            shutil.rmtree(last_model_path, ignore_errors=True)
            shutil.copytree(model_path, last_model_path)
            # Deleting old models
            old_model_path = os.path.join(save_model_path,
                                          f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                          'epoch_{}'.format(epoch_id - 3))
            if os.path.exists(old_model_path):
                shutil.rmtree(old_model_path)
        logger.info('Models have been saved:{}'.format(model_path))

    def __train_epoch(self, epoch_id, local_rank, writer, nranks=0):
        train_times, accuracies, loss_sum = [], [], []
        start = time.time()
        sum_batch = len(self.train_loader) * self.configs.train_conf.max_epoch
        for batch_id, (audio, label, input_lens_ratio) in enumerate(self.train_loader):
            if nranks > 1:
                audio = audio.to(local_rank)
                input_lens_ratio = input_lens_ratio.to(local_rank)
                label = label.to(local_rank).long()
            else:
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
            features, _ = self.audio_featurizer(audio, input_lens_ratio)
            output = self.model(features)
            # Calculation of the value of the loss
            los = self.loss(output, label)
            self.optimizer.zero_grad()
            los.backward()
            self.optimizer.step()

            # Calculation accuracy
            output = torch.nn.functional.softmax(output, dim=-1)
            output = output.data.cpu().numpy()
            output = np.argmax(output, axis=1)
            label = label.data.cpu().numpy()
            acc = np.mean((output == label).astype(int))
            accuracies.append(acc)
            loss_sum.append(los)
            train_times.append((time.time() - start) * 1000)

            # Doka training uses only one process to print
            if batch_id % self.configs.train_conf.log_interval == 0 and local_rank == 0:
                # Calculate the amount of training data per second
                train_speed = self.configs.dataset_conf.batch_size / (sum(train_times) / len(train_times) / 1000)
                # Calculate remaining time
                eta_sec = (sum(train_times) / len(train_times)) * (
                        sum_batch - (epoch_id - 1) * len(self.train_loader) - batch_id)
                eta_str = str(timedelta(seconds=int(eta_sec / 1000)))
                logger.info(f'Train epoch: [{epoch_id}/{self.configs.train_conf.max_epoch}], '
                            f'batch: [{batch_id}/{len(self.train_loader)}], '
                            f'loss: {sum(loss_sum) / len(loss_sum):.5f}, '
                            f'accuracy: {sum(accuracies) / len(accuracies):.5f}, '
                            f'learning rate: {self.scheduler.get_last_lr()[0]:>.8f}, '
                            f'speed: {train_speed:.2f} data/sec, eta: {eta_str}')
                writer.add_scalar('Train/Loss', sum(loss_sum) / len(loss_sum), self.train_step)
                writer.add_scalar('Train/Accuracy', (sum(accuracies) / len(accuracies)), self.train_step)
                # Recording of learning rates
                writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], self.train_step)
                train_times = []
                self.train_step += 1
            start = time.time()
        self.scheduler.step()

    def train(self,
              save_model_path='models/',
              resume_model=None,
              pretrained_model=None,
              augment_conf_path='configs/augmentation.json'):
        """
        Train the model
        :param save_model_path: Path where the model is saved.
        :param resume_model: resume training, when None then no pretrained model is used
        :param pretrained_model: path to pretrained model, when None then no pretrained model is used
        :param augment_conf_path: configuration file for data augmentation, in json format
        """
        # Get training on how many graphics cards there are
        nranks = torch.cuda.device_count()
        local_rank = 0
        writer = None
        if local_rank == 0:
            # logger
            writer = LogWriter(logdir='log')

        if nranks > 1 and self.use_gpu:
            # Initialising the NCCL environment
            dist.init_process_group(backend='nccl')
            local_rank = int(os.environ["LOCAL_RANK"])

        # Getting data
        self.__setup_dataloader(augment_conf_path=augment_conf_path, is_train=True)
        # Getting the model
        self.__setup_model(input_size=self.audio_featurizer.feature_dim, is_train=True)

        # Supports multi-card training
        if nranks > 1 and self.use_gpu:
            self.model.to(local_rank)
            self.audio_featurizer.to(local_rank)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[local_rank])
        logger.info('Training data:{}'.format(len(self.train_dataset)))

        self.__load_pretrained(pretrained_model=pretrained_model)
        # Load recovery model
        last_epoch, best_acc = self.__load_checkpoint(save_model_path=save_model_path, resume_model=resume_model)

        test_step, self.train_step = 0, 0
        last_epoch += 1
        if local_rank == 0:
            writer.add_scalar('Train/lr', self.scheduler.get_last_lr()[0], last_epoch)
        # Start training
        for epoch_id in range(last_epoch, self.configs.train_conf.max_epoch):
            epoch_id += 1
            start_epoch = time.time()
            # Training an epoch
            self.__train_epoch(epoch_id=epoch_id, local_rank=local_rank, writer=writer, nranks=nranks)
            # Multi-card training uses only one process to perform evaluation and save the model
            if local_rank == 0:
                logger.info('=' * 70)
                loss, acc = self.evaluate(resume_model=None)
                logger.info('Test epoch: {}, time/epoch: {}, loss: {:.5f}, accuracy: {:.5f}'.format(
                    epoch_id, str(timedelta(seconds=(time.time() - start_epoch))), loss, acc))
                logger.info('=' * 70)
                writer.add_scalar('Test/Accuracy', acc, test_step)
                writer.add_scalar('Test/Loss', loss, test_step)
                test_step += 1
                self.model.train()
                # # Preserving the optimal model
                if acc >= best_acc:
                    best_acc = acc
                    self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc,
                                           best_model=True)
                # Save the model
                self.__save_checkpoint(save_model_path=save_model_path, epoch_id=epoch_id, best_acc=acc)

    def evaluate(self, resume_model='models/EcapaTdnn_MelSpectrogram/best_model/', save_matrix_path=None):
        """
        Evaluating Models
        :param resume_model: the model used
        :param save_matrix_path: Path to save the mixing matrix
        :return: Evaluation result
        """
        if self.test_loader is None:
            self.__setup_dataloader()
        if self.model is None:
            self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        if resume_model is not None:
            if os.path.isdir(resume_model):
                resume_model = os.path.join(resume_model, 'model.pt')
            assert os.path.exists(resume_model), f"{resume_model} The model does not exist!"
            model_state_dict = torch.load(resume_model)
            self.model.load_state_dict(model_state_dict)
            logger.info(f'Successfully loaded model:{resume_model}')
        self.model.eval()
        if isinstance(self.model, torch.nn.parallel.DistributedDataParallel):
            eval_model = self.model.module
        else:
            eval_model = self.model

        accuracies, losses, preds, labels = [], [], [], []
        with torch.no_grad():
            for batch_id, (audio, label, input_lens_ratio) in enumerate(tqdm(self.test_loader)):
                audio = audio.to(self.device)
                input_lens_ratio = input_lens_ratio.to(self.device)
                label = label.to(self.device).long()
                features, _ = self.audio_featurizer(audio, input_lens_ratio)
                output = eval_model(features)
                los = self.loss(output, label)
                label = label.data.cpu().numpy()
                output = output.data.cpu().numpy()
                # Model prediction labelling
                pred = np.argmax(output, axis=1)
                preds.extend(pred.tolist())
                # truth tag
                labels.extend(label.tolist())

                # Calculation accuracy
                acc = np.mean((pred == label).astype(int))
                accuracies.append(acc)
                losses.append(los.data.cpu().numpy())
        loss = float(sum(losses) / len(losses))
        acc = float(sum(accuracies) / len(accuracies))
        # Save Mixed Matrix
        if save_matrix_path is not None:
            cm = confusion_matrix(labels, preds)
            plot_confusion_matrix(cm=cm, save_path=os.path.join(save_matrix_path, f'{int(time.time())}.png'),
                                  class_labels=self.class_labels)

        self.model.train()
        return loss, acc

    def export(self, save_model_path='models/', resume_model='models/EcapaTdnn_MelSpectrogram/best_model/'):
        """
        Exporting a Predictive Model
        :param save_model_path: Path to where the model is saved
        :param resume_model: Path to the model to be transformed
        :return:
        """
        self.__setup_model(input_size=self.audio_featurizer.feature_dim)
        # Loading pre-trained models
        if os.path.isdir(resume_model):
            resume_model = os.path.join(resume_model, 'model.pt')
        assert os.path.exists(resume_model), f"{resume_model} The model does not exist!"
        model_state_dict = torch.load(resume_model)
        self.model.load_state_dict(model_state_dict)
        logger.info('Successful recovery of model parameters and optimisation method parameters:{}'.format(resume_model))
        self.model.eval()
        # Get static model
        infer_model = self.model.export()
        infer_model_path = os.path.join(save_model_path,
                                        f'{self.configs.use_model}_{self.configs.preprocess_conf.feature_method}',
                                        'inference.pt')
        os.makedirs(os.path.dirname(infer_model_path), exist_ok=True)
        torch.jit.save(infer_model, infer_model_path)
        logger.info("The prediction model has been saved:{}".format(infer_model_path))

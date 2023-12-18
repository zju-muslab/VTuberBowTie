#coding: utf-8

import os
import time
import random
import torch
import torchaudio

import numpy as np
import soundfile as sf
import torch.nn.functional as F

from torch import nn
from torch.utils.data import DataLoader
from librosa.filters import mel as librosa_mel_fn
import librosa
from parallel_wavegan.utils import read_hdf5
import h5py
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

import augment

np.random.seed(1)
random.seed(1)

SPECT_PARAMS = {
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300,
    "power": 1,
}

MEL_PARAMS = {
    "sample_rate": 24000,
    "n_mels": 80,
    "n_fft": 2048,
    "win_length": 1200,
    "hop_length": 300
}

class MelDataset(torch.utils.data.Dataset):
    """
    Dataset class for Mel spectrogram data.
    """

    def __init__(self, data_list, sr=24000, validation=False):
        """
        Initialize the MelDataset.

        Args:
            data_list (list): List of data paths and labels.
            sr (int): Sample rate.
            validation (bool): Flag indicating if the dataset is for validation.
        """
        _data_list = [l[:-1].split('|') for l in data_list]
        self.data_list = [(os.path.join('../..', path), int(label)) for path, label in _data_list]
        self.data_list_per_class = {
            target: [(path, label) for path, label in self.data_list if label == target]
            for target in list(set([label for _, label in self.data_list]))
        }

        self.sr = sr
        self.to_melspec = torchaudio.transforms.MelSpectrogram(**MEL_PARAMS)

        self.mean, self.std = -4, 4

        self.mean = torch.from_numpy(read_hdf5("../../stats/vctk_vocoder_stats.h5", "mean")).float()
        self.std = torch.from_numpy(read_hdf5("../../stats/vctk_vocoder_stats.h5", "scale")).float()

        self.validation = validation
        self.max_mel_length = 192

        self.to_spec = torchaudio.transforms.Spectrogram(**SPECT_PARAMS)

    def __len__(self):
        """
        Get the length of the dataset.

        Returns:
            int: Length of the dataset.
        """
        return len(self.data_list)

    def __getitem__(self, idx):
        """
        Get an item from the dataset.

        Args:
            idx (int): Index of the item.

        Returns:
            tuple: Tuple containing the mel spectrogram, label, reference mel spectrogram, reference 2 mel spectrogram,
                   and reference label.
        """
        data = self.data_list[idx]
        mel_tensor, label = self._load_data(data, sox_augment=True)
        ref_data = random.choice(self.data_list)
        ref_mel_tensor, ref_label = self._load_data(ref_data)
        ref2_data = random.choice(self.data_list_per_class[ref_label])
        ref2_mel_tensor, _ = self._load_data(ref2_data)
        return mel_tensor, label, ref_mel_tensor, ref2_mel_tensor, ref_label

    def _load_data(self, path, sox_augment=False):
        """
        Load data from a file.

        Args:
            path (str): Path to the file.
            sox_augment (bool): Flag indicating if data augmentation using SoX should be applied.

        Returns:
            tuple: Tuple containing the mel spectrogram and label.
        """
        wave_tensor, label = self._load_tensor(path)

        if not self.validation:  # random scale for robustness
            random_scale = 0.5 + 0.5 * np.random.random()
            wave_tensor = random_scale * wave_tensor
            if sox_augment:
                chain = augment.EffectChain()
                if np.random.random() > 0.5:
                    chain = chain.reverb(50, 50, random_room_size).channels()  # spatial
                wave_tensor_ = chain.apply(wave_tensor, src_info={'rate': 24000})
                if not (torch.isnan(wave_tensor_).any() or torch.isinf(wave_tensor_).any()):
                    wave_tensor = wave_tensor_.squeeze()

        mel_tensor = self.log_mel_spectrogram(wave_tensor)
        mel_tensor = self.normalization(mel_tensor)
        mel_length = mel_tensor.size(1)
        if mel_length > self.max_mel_length:
            random_start = np.random.randint(0, mel_length - self.max_mel_length)
            mel_tensor = mel_tensor[:, random_start:random_start + self.max_mel_length]

        return mel_tensor, label

    def normalization(self, mel_tensor):
        """
        Normalize the mel spectrogram.

        Args:
            mel_tensor (torch.Tensor): Mel spectrogram.

        Returns:
            torch.Tensor: Normalized mel spectrogram.
        """
        mel_tensor = mel_tensor.transpose(-1, -2)
        mel_tensor = (mel_tensor - self.mean) / self.std
        mel_tensor = mel_tensor.transpose(-1, -2)
        return mel_tensor

    def log_mel_spectrogram(self, wave):
        """
        Compute the log mel spectrogram.

        Args:
            wave (torch.Tensor): Waveform.

        Returns:
            torch.Tensor: Log mel spectrogram.
        """
        mel_tensor = self.mel_spectrogram(wave, sampling_rate=24000, n_fft=2048, num_mels=80, hop_size=300,
                                          win_size=1200, fmin=80, fmax=7600)
        mel_tensor = torch.clamp(mel_tensor, min=1e-5, max=1000)
        mel_tensor = torch.log10(mel_tensor)
        return mel_tensor

    def mel_spectrogram(self, y, n_fft, num_mels, sampling_rate, hop_size, win_size, fmin, fmax, center=False):
        """
        Compute the mel spectrogram.

        Args:
            y (torch.Tensor): Waveform.
            n_fft (int): FFT window size.
            num_mels (int): Number of mel bins.
            sampling_rate (int): Sampling rate.
            hop_size (int): Hop size.
            win_size (int): Window size.
            fmin (int): Minimum frequency.
            fmax (int): Maximum frequency.
            center (bool): Flag indicating if the window should be centered.

        Returns:
            torch.Tensor: Mel spectrogram.
        """
        if torch.min(y) < -1.:
            print('min value is ', torch.min(y))
        if torch.max(y) > 1.:
            print('max value is ', torch.max(y))

        spec = self.to_spec(y)
        mel_basis = librosa_mel_fn(sampling_rate, n_fft, num_mels, fmin, fmax)
        spec = torch.matmul(torch.from_numpy(mel_basis), spec)
        return spec

    def _load_tensor(self, data):
        """
        Load a tensor from a file.

        Args:
            data (tuple): Tuple containing the file path and label.

        Returns:
            tuple: Tuple containing the tensor and label.
        """
        wave_path, label = data
        label = int(label)
        wave, sr = librosa.load(wave_path, sr=24000)
        wave_tensor = torch.from_numpy(wave).float()
        return wave_tensor, label

class Collater(object):
    """
    Collater class for batch collation.
    """

    def __init__(self, return_wave=False):
        self.text_pad_index = 0
        self.return_wave = return_wave
        self.max_mel_length = 192
        self.mel_length_step = 16
        self.latent_dim = 16

    def __call__(self, batch):
        """
        Collate a batch of data.

        Args:
            batch (list): List of data samples.

        Returns:
            tuple: Tuple containing the collated data.
        """
        batch_size = len(batch)
        nmels = batch[0][0].size(0)
        mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        labels = torch.zeros((batch_size)).long()
        ref_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref2_mels = torch.zeros((batch_size, nmels, self.max_mel_length)).float()
        ref_labels = torch.zeros((batch_size)).long()

        for bid, (mel, label, ref_mel, ref2_mel, ref_label) in enumerate(batch):
            mel_size = mel.size(1)
            mels[bid, :, :mel_size] = mel

            ref_mel_size = ref_mel.size(1)
            ref_mels[bid, :, :ref_mel_size] = ref_mel

            ref2_mel_size = ref2_mel.size(1)
            ref2_mels[bid, :, :ref2_mel_size] = ref2_mel

            labels[bid] = label
            ref_labels[bid] = ref_label

        z_trg = torch.randn(batch_size, self.latent_dim)
        z_trg2 = torch.randn(batch_size, self.latent_dim)

        mels, ref_mels, ref2_mels = mels.unsqueeze(1), ref_mels.unsqueeze(1), ref2_mels.unsqueeze(1)
        return mels, labels, ref_mels, ref2_mels, ref_labels, z_trg, z_trg2

def build_dataloader(path_list, validation=False, batch_size=4, num_workers=1, device='cpu', collate_config={},
                     dataset_config={}):
    """
    Build a data loader.

    Args:
        path_list (list): List of data paths.
        validation (bool): Flag indicating if the data loader is for validation.
        batch_size (int): Batch size.
        num_workers (int): Number of worker processes.
        device (str): Device to use.
        collate_config (dict): Configuration for the collater.
        dataset_config (dict): Configuration for the dataset.

    Returns:
        torch.utils.data.DataLoader: Data loader.
    """
    dataset = MelDataset(path_list, validation=validation, **dataset_config)
    collate_fn = Collater(**collate_config)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=(not validation),
                             num_workers=num_workers,
                             drop_last=(not validation),
                             collate_fn=collate_fn,
                             pin_memory=(device != 'cpu'))

    return data_loader


class ChainRunner:
    """
    Applies an instance of augment.EffectChain on PyTorch tensors.
    """

    def __init__(self, chain):
        self.chain = chain

    def __call__(self, x):
        """
        Apply the effect chain on the input tensor.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        src_info = {
            'channels': x.size(0),  # number of channels
            'length': x.size(1),   # length of the sequence
            'precision': 32,       # precision (16, 32 bits)
            'rate': 16000.0,       # sampling rate
            'bits_per_sample': 32  # size of the sample
        }

        target_info = {
            'channels': 1,
            'length': x.size(1),
            'precision': 32,
            'rate': 16000.0,
            'bits_per_sample': 32
        }

        y = self.chain.apply(x, src_info=src_info, target_info=target_info)

        # SoX might misbehave sometimes by giving NaN/Inf if sequences are too short (or silent)
        # and the effect chain includes eg `pitch`
        if torch.isnan(y).any() or torch.isinf(y).any():
            return x.clone()
        return y


noise_room, sr = torchaudio.load('../../Utils/augment/noise_room.wav')
noise_room = noise_room[0] / torch.mean(torch.abs(noise_room))

# Generate a random shift applied to the speaker's pitch
def random_pitch_shift():
    return np.random.randint(-50, 50)

# Generate a random size of the room
def random_room_size():
    return np.random.randint(0, 50)

def random_bandwidth():
    return np.random.randint(50, 150)

def random_noise_snr():
    return np.random.randint(20, 35)

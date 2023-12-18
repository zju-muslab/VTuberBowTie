import sys
import os
import time
import math
import random
import yaml
import tqdm
from munch import Munch
import numpy as np
import torch
import torchaudio
from torch import nn
import torch.nn.functional as F
import librosa
from librosa.filters import mel as librosa_mel_fn
import matplotlib.pyplot as plt
import copy

sys.path.insert(0, '..')
from parallel_wavegan.utils import load_model
from parallel_wavegan.utils import read_hdf5

class StreamVC():
    def __init__(self, parameters=None):
        # Initialize the StreamVC class with the given parameters
        for key in parameters:
            setattr(self, key, parameters[key])

        # Set the model and config paths
        model_path = os.path.join(self.model_dir, self.model_name)
        config_path = os.path.join(self.model_dir, self.config_name)

        # Add the run directory to the system path
        run_dir = self.run
        sys.path.append(run_dir)
        from models import CRN, MappingNetwork, StyleEncoder

        # Initialize the necessary transforms and models
        self.to_spec = torchaudio.transforms.Spectrogram(n_fft=self.NFFT, win_length=self.NWINDOW, hop_length=self.NHOP, power=1, center=False)
        self.mel_basis = torch.from_numpy(librosa_mel_fn(sr=self.SR, n_fft=self.NFFT, n_mels=self.NMEL, fmin=80, fmax=7600)).to(self.device)

        # Load the CRN model and its components
        with open(config_path) as f:
            CRN_config = yaml.safe_load(f)
        args = Munch(CRN_config["model_params"])
        generator = CRN(args.style_dim, args.F0_channel)
        mapping_network = MappingNetwork(args.latent_dim, args.style_dim, args.num_domains, hidden_dim=args.max_conv_dim)
        style_encoder = StyleEncoder(args.dim_in, args.style_dim, args.num_domains, args.max_conv_dim)
        self.nets = Munch(generator=generator, mapping_network=mapping_network, style_encoder=style_encoder)
        params = torch.load(model_path, map_location='cpu')
        params = params['model']
        _ = [self.nets[key].load_state_dict(params[key]) for key in self.nets]
        _ = [self.nets[key].eval() for key in self.nets]
        self.nets.style_encoder = self.nets.style_encoder.to(self.device)
        self.nets.mapping_network = self.nets.mapping_network.to(self.device)
        self.nets.generator = self.nets.generator.to(self.device)
        self.mean = torch.Tensor(read_hdf5(self.norm_stats_path, "mean")).to(self.device)
        self.std = torch.Tensor(read_hdf5(self.norm_stats_path, "scale")).to(self.device)

        # Load the F0 model
        sys.path.append(os.path.abspath('..'))
        from Utils.pitch_extractor.model import JDCNet
        self.F0_model = JDCNet(num_class=1, seq_len=192).to(self.device)
        params = torch.load(self.f0_path, map_location='cpu')['model']
        self.F0_model.load_state_dict(params)

        # Load the vocoder model
        self.vocoder = load_model(self.vocoder_path).to(self.device)
        self.speakers = [225, 228, 229, 230, 231, 233, 236, 239, 240, 244, 226, 227, 232, 243, 254, 256, 258, 259, 270, 273]
        self.selected_speakers = [228, 230, 233, 236, 243, 244, 254, 258, 259, 273]
        speaker_dicts = {}
        for s in self.selected_speakers:
            speaker_dicts['p' + str(s)] = ('Data/target/p' + str(s) + '/p' + str(s) + '_023.wav', self.speakers.index(s))
        self.reference_embeddings = self.compute_style(self.nets, speaker_dicts)
        self.reference_embeddings_by_index = [self.reference_embeddings[f'p{ref}'][0] for ref in self.selected_speakers]
        self.ref_index = 0

        self.pad = torch.nn.ReplicationPad2d((2, 2, 0, 0))
        self.receive_buff = []
        self.is_speeches = []
        self.f0_conv_state = None
        self.restart()
        self.STFT_OVERLAP_SIZE = self.NFFT - self.NHOP
        self.stft_buff = np.zeros(self.STFT_OVERLAP_SIZE)
        self.vad, utils = torch.hub.load(repo_or_dir=self.vad_path, source='local', model='silero_vad')
        self.vad = self.vad.to(self.device)
        self.resample = torchaudio.transforms.Resample(orig_freq=24000, new_freq=8000).to(self.device)
        self.is_convert = False

    def feed(self, frame):
        # Cache the frame for feature extraction
        frame = np.concatenate((self.stft_buff, frame), axis=-1)
        self.stft_buff = frame[..., -self.STFT_OVERLAP_SIZE:]
        frame = torch.from_numpy(frame).float().to(self.device)

        self.is_convert = True
        with torch.no_grad():
            is_speech = self.is_active(frame.reshape(-1))
            self.is_speeches.append(is_speech)
            mel_in = self.preprocess(frame).unsqueeze(1)
            f0 = self.f0_feed(mel_in)
            mel_out = self.generator_feed(mel_in, self.reference_embeddings_by_index[self.ref_index], f0, is_speech=is_speech)
            m, s = self.mean.reshape(1, 1, -1, 1), self.std.reshape(1, 1, -1, 1)
            m_, s_ = self.vocoder.mean.reshape(1, 1, -1, 1), self.vocoder.scale.reshape(1, 1, -1, 1)
            mel_out = ((mel_out * s + m) - m_) / s_
            y = self.vocoder_feed(mel_out)
        self.is_convert = False
        return y

    def convert(self, x, ref_index):
        with torch.no_grad():
            ref = self.reference_embeddings_by_index[ref_index]
            x = self.preprocess(x).squeeze().unsqueeze(0).unsqueeze(0).to(self.device)
            F0_feature = self.F0_model.get_feature_GAN(x)
            mel_out, _, hc = self.nets.generator(x, ref, F0=F0_feature)
            mel_out = mel_out.transpose(-1, -2).squeeze()
            mel_out = ((mel_out * self.std + self.mean) - self.vocoder.mean) / self.vocoder.scale
            y_out = self.vocoder.inference(mel_out).view(-1).cpu()
        return y_out

    def convert_live_simulate(self, x, ref_index):
        if ref_index != self.ref_index:
            self.change_speaker(ref_index)
        else:
            self.restart()

        chunk_size = self.CHUNK
        y_out = []
        cnt = 1
        while x.shape[-1] > (cnt) * chunk_size:
            x_in = x[..., (cnt - 1) * chunk_size:cnt * chunk_size]
            _y_out = self.feed(x_in)
            y_out.append(_y_out.reshape(-1))
            cnt += 1
        y_out = torch.stack(y_out).reshape(-1)
        return y_out

    def change_speaker(self, ref_index):
        while self.is_convert:
            time.sleep(0.001)
        self.restart()
        self.ref_index = ref_index
        print(f'---change speaker to p{self.selected_speakers[self.ref_index]} ')
        return

    def restart(self):
        self.conv_state = None
        self.lstm_state = None
        self.norm_state = None
        self.f0_conv_state = None
        self.vocoder_conv_state = None

    def vocoder_feed(self, x):
        out, self.vocoder_conv_state = self.vocoder.feed(x, self.vocoder_conv_state)
        return out

    def f0_feed(self, x):
        F0_feature, self.f0_conv_state = self.F0_model.feed(x, self.f0_conv_state)
        return F0_feature

    def generator_feed(self, x, ref, F0, is_speech=False):
        out, self.conv_state, self.norm_state, self.lstm_state = self.nets.generator.feed(x, ref, F0, self.conv_state, self.norm_state, self.lstm_state, use_input_stats=True, update_mean_var=False, is_speech=is_speech)
        return out

    def is_active(self, x, threshold=0.2, sample_rate=8000):
        x = self.resample(x)
        with torch.no_grad():
            speech_prob = self.vad(x, sample_rate).item()
        return speech_prob > threshold

    def compute_style(self, nets, speaker_dicts):
        reference_embeddings = {}
        for key, (path, speaker) in speaker_dicts.items():
            if path == "":
                label = torch.LongTensor([speaker]).to(self.device)
                latent_dim = nets.mapping_network.shared[0].in_features
                ref = nets.mapping_network(torch.randn(1, latent_dim).to(self.device), label)
            else:
                wave, sr = librosa.load(path, sr=24000)
                if sr != 24000:
                    wave = librosa.resample(wave, sr, 24000)
                mel_tensor = self.preprocess(wave).to(self.device)

                with torch.no_grad():
                    label = torch.LongTensor([speaker])
                    ref = nets.style_encoder(mel_tensor.unsqueeze(1), label)
            reference_embeddings[key] = (ref, label)
        return reference_embeddings

    def preprocess(self, wave, new_feature=True):
        if isinstance(wave, np.ndarray):
            wave = torch.from_numpy(wave).float().to(self.device)
        mel_tensor = self._mel_spectrogram(wave)
        mel_tensor = torch.clamp(mel_tensor, min=1e-10, max=1000).transpose(-1, -2)
        mel_tensor = torch.log10(mel_tensor)
        mel_tensor = ((mel_tensor - self.mean) / self.std)
        mel_tensor = mel_tensor.transpose(-1, -2)
        return mel_tensor.unsqueeze(0)

    def _mel_spectrogram(self, y):
        if torch.max(torch.abs(y)) > 1:
            y = y / (torch.max(torch.abs(y)) + 1e-5)
        spec = self.to_spec(y.cpu()).to(self.device)
        spec = torch.mm(self.mel_basis, spec)
        return spec

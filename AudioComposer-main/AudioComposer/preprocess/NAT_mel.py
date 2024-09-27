import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from scipy.io.wavfile import read
import torch
import torch.nn as nn

MAX_WAV_VALUE = 32768.0


def load_wav(full_path):
    sampling_rate, data = read(full_path)
    return data, sampling_rate


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log10(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log10(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output


def spectral_de_normalize_torch(magnitudes):
    output = dynamic_range_decompression_torch(magnitudes)
    return output

class MelNet(nn.Module):
    def __init__(self,hparams,device='cpu') -> None:
        super().__init__()
        self.n_fft = hparams['fft_size']
        self.num_mels = hparams['audio_num_mel_bins']
        self.sampling_rate = hparams['audio_sample_rate']
        self.hop_size = hparams['hop_size']
        self.win_size = hparams['win_size']
        self.fmin = hparams['fmin']
        self.fmax = hparams['fmax']
        self.device = device
        
        mel = librosa_mel_fn(self.sampling_rate, self.n_fft, self.num_mels, self.fmin, self.fmax)
        self.mel_basis = torch.from_numpy(mel).float().to(self.device)
        self.hann_window = torch.hann_window(self.win_size).to(self.device)

    def to(self,device,**kwagrs):
        super().to(device=device,**kwagrs)
        self.mel_basis = self.mel_basis.to(device)
        self.hann_window = self.hann_window.to(device)
        self.device = device

    def forward(self,y,center=False, complex=False):
        if isinstance(y,np.ndarray):
            y = torch.FloatTensor(y)
            if len(y.shape) == 1:
                y = y.unsqueeze(0)
        y = y.clamp(min=-1., max=1.).to(self.device)

        y = torch.nn.functional.pad(y.unsqueeze(1), [int((self.n_fft - self.hop_size) / 2), int((self.n_fft - self.hop_size) / 2)],
                                    mode='reflect')
        y = y.squeeze(1)

        spec = torch.stft(y, self.n_fft, hop_length=self.hop_size, win_length=self.win_size, window=self.hann_window,
                        center=center, pad_mode='reflect', normalized=False, onesided=True,return_complex=complex)

        if not complex:
            spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
            spec = torch.matmul(self.mel_basis, spec)
            spec = spectral_normalize_torch(spec)
        else:
            B, C, T, _ = spec.shape
            spec = spec.transpose(1, 2)  # [B, T, n_fft, 2]
        return spec

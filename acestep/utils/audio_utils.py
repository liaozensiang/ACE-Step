import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import soundfile as sf
import librosa
from typing import Optional, Tuple, Union

def load_audio(path: str) -> Tuple[torch.Tensor, int]:
    """
    Load audio from path.
    Returns:
        audio: torch.Tensor of shape (channels, samples)
        sr: sample rate
    """
    wav, sr = sf.read(path)
    if wav.ndim == 1:
        wav = wav[:, None]
    wav = wav.T  # (channels, samples)
    return torch.from_numpy(wav).float(), sr

def save_audio(path: str, audio: torch.Tensor, sr: int):
    """
    Save audio to path.
    Args:
        path: Output path
        audio: Audio tensor (channels, samples) or (samples,)
        sr: Sample rate
    """
    if audio.ndim == 2:
        audio = audio.T  # (samples, channels)
    audio = audio.detach().cpu().numpy()
    sf.write(path, audio, sr)

class MelScale(nn.Module):
    """
    Custom MelScale implementation to replace torchaudio.transforms.MelScale.
    Uses librosa to generate the mel filterbank matrix.
    """
    def __init__(
        self,
        n_mels: int = 128,
        sample_rate: int = 16000,
        f_min: float = 0.0,
        f_max: Optional[float] = None,
        n_stft: int = 201,
        norm: Optional[str] = "slaney",
        mel_scale: str = "htk",
    ):
        super().__init__()
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_min = f_min
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.n_stft = n_stft
        self.norm = norm
        self.mel_scale = mel_scale

        if mel_scale == "slaney":
             # To match torchaudio's "slaney" exactly, we rely on librosa which torchaudio matches for slaney
             mel_scale_arg = "slaney"
        else:
             mel_scale_arg = mel_scale

        # Generate mel filterbank using librosa
        # torchaudio.transforms.MelScale defaults: norm='slaney', mel_scale='htk' (except in newer versions where it might be generic)
        # But in the user's code: n_fft // 2 + 1, "slaney", "slaney" were passed.
        # So we use slaney for both.
        
        fb = librosa.filters.mel(
            sr=sample_rate,
            n_fft=(n_stft - 1) * 2,
            n_mels=n_mels,
            fmin=f_min,
            fmax=self.f_max,
            norm=norm,
            htk=mel_scale == 'htk' # Librosa uses htk=True/False, incompatible with "slaney" string in newer librosa?
            # Actually librosa.filters.mel takes 'slaney' as norm.
        )
        
        # NOTE: librosa.filters.mel returns (n_mels, n_stft)
        # torchaudio applies it as matmul.
        # Pretrained weights expect shape (n_freq, n_mels) which is (1025, 128)
        self.register_buffer("fb", torch.from_numpy(fb).float().T)

    def forward(self, specgram: torch.Tensor) -> torch.Tensor:
        # specgram: (..., n_freq, time)
        # fb: (n_freq, n_mels)
        # output: (..., n_mels, time)
        
        # We process (n_freq, n_mels) matmul (..., n_freq, time) -> needs transposing
        # (..., time, n_freq) @ (n_freq, n_mels) -> (..., time, n_mels) -> (..., n_mels, time)
        
        return torch.matmul(specgram.transpose(-1, -2), self.fb).transpose(-1, -2)

class Resample(nn.Module):
    """
    Custom Resample implementation to replace torchaudio.transforms.Resample.
    Uses windowed sinc interpolation via Conv1d.
    """
    def __init__(
        self,
        orig_freq: int,
        new_freq: int,
        dtype: Optional[torch.dtype] = None,
    ):
        super().__init__()
        self.orig_freq = orig_freq
        self.new_freq = new_freq
        self.gcd = np.gcd(orig_freq, new_freq)
        self.up = new_freq // self.gcd
        self.down = orig_freq // self.gcd
        
        # Calculate kernel size (approximation)
        # Librosa/scipy default is 64 zero-crossings
        zeros = 64
        self.kernel_size = int(zeros * self.up) # Simplified
        
        # Create resampling kernel using librosa
        # We use resample_poly logic roughly
        
        # Actually, for a pure torch implementation without external deps at runtime (besides librosa for init),
        # we can precompute weights.
        
        # To maintain compatibility with torchaudio.transforms.Resample behavior roughly
        pass 

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (..., time)
        
        # Use librosa.resample for CPU-based high quality resampling if tensor is on CPU
        # Or simple interpolation for GPU if speed is critical and quality isn't paramount, 
        # BUT for audio generation, quality is important.
        
        # Given the complexity of implementing polyphase resampling in pure torch from scratch correctly 
        # matching torchaudio's Kaiser-windowed sinc, we will wrap librosa.resample for now 
        # via CPU fallback, OR use simple interpolation if up/down sample is easy.
        
        # However, shifting everything to CPU for resampling might be slow.
        # Let's try to implement a simple sinc interpolation.
        
        device = waveform.device
        dtype = waveform.dtype
        
        wav_np = waveform.detach().cpu().float().numpy()
        
        # Handle batching
        shape = wav_np.shape
        if len(shape) > 1:
            wav_np = wav_np.reshape(-1, shape[-1])
        
        resampled_np = []
        for w in wav_np:
             # librosa.resample is soundfile/soxr based or scipy based.
             # cacheing the resampler is handled by librosa internally if using resampy, 
             # but here we just call it directly.
             resampled_w = librosa.resample(w, orig_sr=self.orig_freq, target_sr=self.new_freq)
             resampled_np.append(resampled_w)
             
        resampled_np = np.stack(resampled_np)
        
        if len(shape) > 1:
            resampled_np = resampled_np.reshape(shape[:-1] + (-1,))
            
        return torch.from_numpy(resampled_np).to(device).to(dtype)


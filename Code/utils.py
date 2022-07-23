import torch
import soundfile as sf
from torch import nn
import numpy as np


EPSILON = 1e-2

def linear_quantize(samples, q_levels):
    samples = samples.clone()
    samples -= samples.min(dim=-1)[0].expand_as(samples)
    samples /= samples.max(dim=-1)[0].expand_as(samples)
    samples *= q_levels - EPSILON
    samples += EPSILON / 2
    return samples.long()

def linear_dequantize(samples, q_levels):
    return samples.float() / (q_levels / 2) - 1

def q_zero(q_levels):
    return q_levels // 2

"""TBD mu-law quantization"""
def mu_law_quantization(audio_samples, q_levels):

    # Encode the samples

    mu = q_levels - 1
    safe_audio_abs = torch.min(torch.abs(audio_samples),torch.as_tensor(1.0))
    magnitude = (torch.log(1 + torch.as_tensor(mu)*(safe_audio_abs)))/(torch.log(1 + torch.as_tensor(mu)))
    samples = torch.sign(audio_samples)*magnitude
    samples = (samples + 1)/2*mu + 0.5
    quantized_samples = samples.type(torch.int32)

    return quantized_samples

def mu_law_dequantization(samples, q_levels):
    mu = q_levels - 1
    # Map values between -1 to 1
    casted =  samples.type(torch.float32)
    samples = 2 * (casted / mu) - 1

    # inverse mu-law quantization
    magnitude = (1 / mu)*(( 1 + mu)**abs(samples) - 1)
    return torch.sign(samples) * magnitude

def write_wav(path, audio, sample_rate):
    sf.write(path, np.array(audio), sample_rate)
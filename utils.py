import numpy as np
from scipy.io.wavfile import read
import torch
import librosa


def get_mask_from_lengths(lengths):
    max_len = torch.max(lengths)
    ids = torch.arange(0, max_len).long().cuda()
    mask = (ids < lengths.unsqueeze(1)).byte()
    return mask


def load_wav_to_torch(full_path, sr):
    sampling_rate, data = read(full_path)
    assert sr == sampling_rate, "{} SR doesn't match {} on path {}".format(
        sr, sampling_rate, full_path)
    return torch.FloatTensor(data.astype(np.float32))


def load_filepaths_and_text(filename, sort_by_length, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]

    if sort_by_length:
        filepaths_and_text.sort(key=lambda x: len(x[1]))

    return filepaths_and_text


def to_gpu(x):
    x = x.contiguous().cuda(async=True)
    return torch.autograd.Variable(x)


def get_split_mels(splited_audios, sr=8000, n_fft=512, win_length=0.025, hop_length=0.01, mel=40):
    log_mels = []
    for audio in splited_audios:
        S = librosa.core.stft(y=audio, n_fft=n_fft, win_length=int(win_length * sr), hop_length=int(sr * hop_length))
        S = np.abs(S) ** 2
        mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=mel)
        S = np.log10(np.dot(mel_basis, S) + 1e-6)
        log_mels.append(S)
    return log_mels


def split_audio(x, sr=22050, seg_length=0.8, pad=False):
    l = x.shape[0] / sr
    L = int(l / seg_length)
    audio_list = []
    for i in range(L - 1):
        audio_list.append(x[int(i * seg_length * sr):int((i + 2) * seg_length * sr)])
    return audio_list

import random

import librosa
import numpy as np
import torch
import torch.utils.data
import audio

import layers
from utils import load_wav_to_torch, load_filepaths_and_text, get_split_mels, split_audio
from text import text_to_sequence

_pad = 0


class TextMelLoader(torch.utils.data.Dataset):
    """
        1) loads audio,text pairs
        2) normalizes text and converts them to sequences of one-hot vectors
        3) computes mel-spectrograms from audio files.
    """

    # pair ==>[]
    def __init__(self, audiopaths_and_text, hparams, shuffle=True):
        self.audiopaths_and_text = load_filepaths_and_text(
            audiopaths_and_text, hparams.sort_by_length)
        self.text_cleaners = hparams.text_cleaners
        self.max_wav_value = hparams.max_wav_value
        self.sampling_rate = hparams.sampling_rate
        self.load_mel_from_disk = hparams.load_mel_from_disk
        self.speaker_encoder = layers.SpeakerEncoder(hparams.num_mel, )
        self.speaker_encoder.load_model(hparams.se_checkpoint)
        self.speaker_encoder.eval()
        self.hparms = hparams
        self.stft = layers.TacotronSTFT(
            hparams.filter_length, hparams.hop_length, hparams.win_length,
            hparams.n_mel_channels, hparams.sampling_rate, hparams.mel_fmin,
            hparams.mel_fmax)
        random.seed(1234)
        if shuffle:
            random.shuffle(self.audiopaths_and_text)

    def get_mel_text_pair(self, audiopath_and_text):
        # separate filename and text
        audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
        text = self.get_text(text)
        speaker_encoder, spectrum, mel = self.get_mel(audiopath)
        return (text, mel, spectrum, speaker_encoder)

    def get_mel(self, filename):
        if not self.load_mel_from_disk:
            wav,_ = librosa.load(filename, self.sampling_rate)
            wav = torch.from_numpy(wav).float().unsqueeze(0)
            #audio_norm = wav / self.max_wav_value
            #audio_norm = audio_norm.unsqueeze(0)
            audio_norm = torch.autograd.Variable(wav, requires_grad=False)
            melspec = self.stft.mel_spectrogram(audio_norm)
            melspec = torch.squeeze(melspec, 0)
            wav, sr = librosa.load(filename, sr=self.hparms.se_sample_rate)
            wav, _ = librosa.effects.trim(wav, top_db=20)
            audios = split_audio(wav, sr=self.hparms.se_sample_rate, )
            mels = get_split_mels(audios,
                                  # sr=self.hparms.se_sample_rate,
                                  # n_fft=self.hparms.se_n_fft,
                                  # win_length=self.hparms.se_window,
                                  # hop_length=self.hparms.se_hop,
                                  mel=self.hparms.num_mel)
            if len(mels)==0:
                print(filename)

            mels = np.stack(mels)
            mels = torch.from_numpy(mels).float()
            mels = mels.permute(0, 2, 1)
            x, _ = self.speaker_encoder(mels, return_sim=False)
            speaker_encoder = x.mean(0)  # final speaker encode from an audio
            # reference from gst
            spectrogram = audio.spectrogram(wav).astype(np.float32)
            spectrogram = spectrogram.transpose(1,0)
        else:
            melspec = torch.from_numpy(np.load(filename))
            assert melspec.size(0) == self.stft.n_mel_channels, (
                'Mel dimension mismatch: given {}, expected {}'.format(
                    melspec.size(0), self.stft.n_mel_channels))

        return speaker_encoder, spectrogram, melspec

    def get_text(self, text):
        text_norm = torch.IntTensor(text_to_sequence(text, self.text_cleaners))
        return text_norm

    def __getitem__(self, index):
        return self.get_mel_text_pair(self.audiopaths_and_text[index])

    def __len__(self):
        return len(self.audiopaths_and_text)


class TextMelCollate():
    """ Zero-pads model inputs and targets based on number of frames per setep
    """

    def __init__(self, n_frames_per_step):
        self.n_frames_per_step = n_frames_per_step

    def __call__(self, batch):
        """Collate's training batch from normalized text and mel-spectrogram,
        spectrum and 256-dim speaker_encoder
        PARAMS
        ------
        batch: [text_normalized, mel_normalized]
        """
        # Right zero-pad all one-hot text sequences to max input length
        input_lengths, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([len(x[0]) for x in batch]),
            dim=0, descending=True)
        max_input_len = input_lengths[0]

        text_padded = torch.LongTensor(len(batch), max_input_len)
        text_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            text = batch[ids_sorted_decreasing[i]][0]
            text_padded[i, :text.size(0)] = text

        spectrum = _prepare_targets([x[2] for x in batch], self.n_frames_per_step)
        d_vector = [x[3] for x in batch]
        d_vector = torch.stack(d_vector)
        # Right zero-pad mel-spec with extra single zero vector to mark the end
        num_mels = batch[0][1].size(0)
        max_target_len = max([x[1].size(1) for x in batch]) + 1
        if max_target_len % self.n_frames_per_step != 0:
            max_target_len += self.n_frames_per_step - max_target_len % self.n_frames_per_step
            assert max_target_len % self.n_frames_per_step == 0

        # include mel padded and gate padded
        mel_padded = torch.FloatTensor(len(batch), num_mels, max_target_len)
        mel_padded.zero_()
        gate_padded = torch.FloatTensor(len(batch), max_target_len)
        gate_padded.zero_()
        output_lengths = torch.LongTensor(len(batch))
        for i in range(len(ids_sorted_decreasing)):
            mel = batch[ids_sorted_decreasing[i]][1]
            mel_padded[i, :, :mel.size(1)] = mel
            gate_padded[i, mel.size(1):] = 1
            output_lengths[i] = mel.size(1)

        return text_padded, input_lengths, mel_padded, gate_padded, output_lengths, spectrum, d_vector


def _prepare_targets(targets, alignment):
    max_len = max((t.shape[0] for t in targets)) + 1
    return np.stack([_pad_target(t, _round_up(max_len, alignment)) for t in targets])


def _pad_input(x, length):
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def _pad_target(t, length):
    return np.pad(t, [(0, length - t.shape[0]), (0, 0)], mode='constant', constant_values=_pad)


def _round_up(x, multiple):
    remainder = x % multiple
    return x if remainder == 0 else x + multiple - remainder

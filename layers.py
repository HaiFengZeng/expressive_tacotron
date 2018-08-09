import torch
from librosa.filters import mel as librosa_mel_fn
from audio_processing import dynamic_range_compression
from audio_processing import dynamic_range_decompression
from stft import STFT
from ge2e_hparams import hparams
from torch import nn
from torch.nn import init
import torch.functional as F


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert (kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length=1024, hop_length=256, win_length=1024,
                 n_mel_channels=80, sampling_rate=22050, mel_fmin=0.0,
                 mel_fmax=None):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        """Computes mel-spectrograms from a batch of waves
        PARAMS
        ------
        y: Variable(torch.FloatTensor) with shape (B, T) in range [-1, 1]

        RETURNS
        -------
        mel_output: torch.FloatTensor of shape (B, n_mel_channels, T)
        """
        assert (torch.min(y.data) >= -1)
        assert (torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        return mel_output


class SpeakerEncoder(nn.Module):
    def __init__(self, input_size, n=hparams.N, m=hparams.M, hidden_size=768, project_size=256):
        super(SpeakerEncoder, self).__init__()
        self.w = nn.Parameter(torch.tensor(10.0))
        self.b = nn.Parameter(torch.tensor(-5.0))
        self.N = n
        self.M = m
        if hparams.mode == 'TD-SV':
            hidden_size = hparams.hidden_size_tdsv
            project_size = hparams.project_size_tdsv
        else:
            hidden_size = hparams.hidden_size_tisv
            project_size = hparams.project_size_tisv
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project1 = nn.Linear(hidden_size, project_size)
        self.lstm2 = nn.LSTM(input_size=project_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project2 = nn.Linear(hidden_size, project_size)
        self.lstm3 = nn.LSTM(input_size=project_size, hidden_size=hidden_size, dropout=0.5,
                             batch_first=False)
        self.project3 = nn.Linear(hidden_size, project_size)
        # self.bn1 = nn.BatchNorm1d(hidden_size)
        self.init()

    def init_lstm(self, lstm):
        for layer in lstm.all_weights:
            for p in layer:
                if len(p.size()) >= 2:
                    init.orthogonal_(p)

    def init(self):
        self.init_lstm(self.lstm1)
        self.init_lstm(self.lstm2)
        self.init_lstm(self.lstm3)
        init.normal_(self.project1.weight.data, 0, 0.02)
        init.normal_(self.project2.weight.data, 0, 0.02)
        init.normal_(self.project3.weight.data, 0, 0.02)

    def similarity_matrix(self, x):
        N, M = self.N, self.M
        # x [N*M,d] B=N*M,d is a vector
        yy = x.unsqueeze(0).repeat(N, 1, 1)
        c = torch.stack(x.split([M] * N), 0).mean(1, keepdim=True)
        cc = c.repeat(1, M * N, 1)
        cc = cc.permute(1, 0, 2)
        yy = yy.permute(1, 0, 2)
        sim = F.cosine_similarity(cc, yy, dim=-1)
        similarity = self.w * sim + self.b
        return similarity

    def forward(self, x, return_sim=True):

        x, (h1, c1) = self.lstm1(x)
        x = x.permute(1, 0, 2)
        x = self.project1(x)
        x = x.permute(1, 0, 2)
        x, (h2, c2) = self.lstm2(x)
        x = x.permute(1, 0, 2)
        x = self.project2(x)
        x = x.permute(1, 0, 2)
        x, (h3, c3) = self.lstm3(x)
        x = x.permute(1, 0, 2)
        x = self.project3(x)
        x = x.permute(1, 0, 2)
        x = x[-1, :, :]
        # l2 norm
        x = x / torch.norm(x)
        if not return_sim:
            return x, None
        sim = self.similarity_matrix(x)
        return x, sim

    def load_model(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        self.load_state_dict(checkpoint['state_dict'])

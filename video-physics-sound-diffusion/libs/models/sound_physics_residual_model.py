import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms
import julius
import librosa
import transformer_modules as tm
import math

class SpecTransformer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(SpecTransformer, self).__init__()
        self.input_size = in_channels
        self.output_size = out_channels
        self.block_size = 44
        self.n_head = 4
        self.n_embd = middle_channels
        self.n_layer = 4
        self.embd_pdrop = 0.1
        self.resid_pdrop = 0.1
        self.attn_pdrop = 0.1
        self.transformer = nn.ModuleDict(dict(
            wte=nn.Linear(self.input_size, self.n_embd),
            wpe=nn.Embedding(self.block_size, self.n_embd),
            drop=nn.Dropout(self.embd_pdrop),
            h=nn.ModuleList([tm.Block(self.n_embd, self.n_head, self.attn_pdrop, self.resid_pdrop) for _ in range(self.n_layer)]),
            ln_f=nn.LayerNorm(self.n_embd),
        ))
        self.lm_head = nn.Linear(self.n_embd, self.output_size, bias=False)

        # init all weights, and apply a special scaled init to the residual projections, per GPT-2 paper
        self.apply(self._init_weights)
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * self.n_layer))

        # report number of parameters (note we don't count the decoder parameters in lm_head)
        n_params = sum(p.numel() for p in self.transformer.parameters())
        print("number of parameters: %.2fM" % (n_params / 1e6,))
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

    def forward(self, spec):
        device = spec.device
        b, t, f = spec.size()
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0)  # shape (1, t)
        # forward the GPT model itself
        tok_emb = self.transformer.wte(spec)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (1, t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)

        return x, logits
class MLP(nn.Module):
    def __init__(self, in_channels, channels):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, channels)
        )
    def forward(self, x):
        return self.mlp(x)

class SpecEncoder(nn.Module):
    def __init__(self, in_channels, channels):
        super(SpecEncoder, self).__init__()
        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(in_channels if i == 0 else channels,
                         channels,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(channels))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)
    def forward(self, x):
        for conv in self.convolutions:
            x = F.leaky_relu(conv(x))
        z = F.adaptive_avg_pool1d(x, 1).squeeze(2)
        return z, x

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
            assert(kernel_size % 2 == 1)
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

class Decoder(nn.Module):
    """Decoder module:
    """

    def __init__(self, input_dim, hidden_dim, out_dim):
        super(Decoder, self).__init__()

        self.lstm1 = nn.LSTM(input_dim, hidden_dim, 1, batch_first=True)

        convolutions = []
        for i in range(3):
            conv_layer = nn.Sequential(
                ConvNorm(hidden_dim,
                         hidden_dim,
                         kernel_size=5, stride=1,
                         padding=2,
                         dilation=1, w_init_gain='relu'),
                nn.BatchNorm1d(hidden_dim))
            convolutions.append(conv_layer)
        self.convolutions = nn.ModuleList(convolutions)

        self.lstm2 = nn.LSTM(hidden_dim, out_dim, 2, batch_first=True)

        self.linear_projection = LinearNorm(out_dim, out_dim)
        # self.drop1 = nn.Dropout(p=0.2)

    def forward(self, x):

        # self.lstm1.flatten_parameters()
        x, _ = self.lstm1(x)
        x = x.transpose(1, 2)

        for conv in self.convolutions:
            x = F.relu(conv(x))
        x = x.transpose(1, 2)

        outputs, _ = self.lstm2(x)
        decoder_output = self.linear_projection(outputs)
        # decoder_output = self.drop1(self.linear_projection(outputs))

        return decoder_output

class ImpactSound(nn.Module):
    def __init__(self, input_dim=1025, enc_audio_fea_dim=256, dec_audio_fea_dim=256,
                 output_dim=1025, n_bins=100, low_dim=256, sr=44100, dur=0.25) -> None:
        super().__init__()
        self.sr = sr
        self.dur = dur
        self.n_bins = n_bins
        self.audio_enc = SpecTransformer(input_dim, enc_audio_fea_dim, low_dim)
        self.noise_proj = MLP(enc_audio_fea_dim, 100)
        self.noise_t_proj = MLP(enc_audio_fea_dim, 100)
        mLen = int(self.sr * self.dur)
        self.tt = torch.arange(0, mLen, dtype=torch.float32)/self.sr
        self.eps = 1e-8
        torch.manual_seed(0)
        noise = torch.randn(1, 11025)
        self.split_bands = julius.SplitBands(sample_rate=sr, n_bands=100)
        self.noise = self.split_bands(noise).squeeze(1)
    def forward(self, gt_spec, pred_wav):
        # input B, D, T
        B, _, _ = gt_spec.shape
        audio_fea, res_audio_fea = self.audio_enc(gt_spec.transpose(1, 2)) # B,T,F
        audio_fea = F.adaptive_avg_pool1d(audio_fea.transpose(1, 2), 1).squeeze(2)
        noise_weights = torch.sigmoid(self.noise_proj(audio_fea))
        noise_t = (1e-5+torch.sigmoid(self.noise_t_proj(audio_fea))*0.5)
        # B, D
        pre_weight_noise = self.noise.to(gt_spec.device).unsqueeze(0).expand(B, 100, -1)
        noise_dcy = self.tt.to(gt_spec.device).unsqueeze(0).unsqueeze(0).expand(B, noise_t.shape[1], -1) * (60 / (noise_t.unsqueeze(2)))
        noise_env = 10 ** (-noise_dcy / 20)
        weighted_noise = torch.sum(noise_weights.unsqueeze(2)*pre_weight_noise*noise_env, dim=1)
        wav = pred_wav + weighted_noise
        wav_ = wav/torch.max(torch.abs(wav), 1, keepdim=True)[0]
        return wav_

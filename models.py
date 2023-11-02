import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

CFG = {
    #'block_size': 17280,
    #'block_stride': 17280 // 16,
    #'patch_size': 12,
    'block_size': 18000,
    'block_stride': 18000 // 8,
    'patch_size': 24,

    'sleepformer_dim': 192,
    'sleepformer_num_heads': 6,
    'sleepformer_num_encoder_layers': 5,
    'sleepformer_num_lstm_layers': 2,
    'sleepformer_lstm_dropout': 0.0,
    'sleepformer_first_dropout': 0.1,
    'sleepformer_encoder_dropout': 0.1,
    'sleepformer_mha_dropout': 0.0,
    'sleepformer_ffn_multiplier': 1,

}


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len):
        super(PositionalEncoding, self).__init__()
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, dim))

    def forward(self, x, training):
        """if training:
            random_shifts = torch.randint(-x.size(1), x.size(1), (x.size(0),), device=x.device)
            x_shifted = torch.stack([x[i].roll(shifts=random_shifts[i].item(), dims=0) for i in range(x.size(0))])
            return x_shifted + self.pos_encoding
        else:
            return x + self.pos_encoding"""
        return x + self.pos_encoding
    



class EncoderLayer(nn.Module):
    def __init__(self, CFG):
        super(EncoderLayer, self).__init__()
        self.CFG = CFG

        self.mha = nn.MultiheadAttention(self.CFG['sleepformer_dim'], self.CFG['sleepformer_num_heads'])

        self.layer_norm1 = nn.LayerNorm(self.CFG['sleepformer_dim'])
        self.layer_norm2 = nn.LayerNorm(self.CFG['sleepformer_dim'])

        self.sequential = nn.Sequential(
            nn.Linear(self.CFG['sleepformer_dim'], self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier']),
            nn.GELU(),
            nn.Linear(self.CFG['sleepformer_dim']*self.CFG['sleepformer_ffn_multiplier'], self.CFG['sleepformer_dim']),
            nn.Dropout(self.CFG['sleepformer_first_dropout'])
        )

        self.mha_dropout = nn.Dropout(self.CFG['sleepformer_mha_dropout'])
        self.attention_weights = None

    def forward(self, x):
        x_norm = self.layer_norm1(x)
        attn_output, att_weights = self.mha(x_norm, x_norm, x_norm)
        self.attention_weights = att_weights
        x = x + self.mha_dropout(attn_output)

        x = x + self.sequential(self.layer_norm2(x))

        return x


class SleepformerEncoder(nn.Module):
    def __init__(self, CFG):
        super(SleepformerEncoder, self).__init__()
        self.CFG = CFG

        self.pos_enc = PositionalEncoding(self.CFG['sleepformer_dim'], self.CFG['block_size'] // self.CFG['patch_size'])
        self.first_dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.enc_layers = nn.ModuleList([EncoderLayer(self.CFG) for _ in range(self.CFG['sleepformer_num_encoder_layers'])])
        self.first_linear = nn.Linear(self.CFG['patch_size']*3, self.CFG['sleepformer_dim'])

        self.lstm_layers = nn.ModuleList()
        for i in range(self.CFG['sleepformer_num_lstm_layers']):
            input_dim = self.CFG['sleepformer_dim'] * (2 if i > 0 else 1)
            self.lstm_layers.append(nn.LSTM(input_dim, self.CFG['sleepformer_dim'],
                                            bidirectional=True, batch_first=True, dropout=self.CFG['sleepformer_lstm_dropout'] if i < self.CFG['sleepformer_num_lstm_layers'] - 1 else 0))

    def forward(self, x, training=True):
        # Try normalise x maybe? Data should already be normalized at this point though
        x = self.first_linear(x)

        x = self.pos_enc(x, training=training)


        x = self.first_dropout(x)
        x = x.transpose(0, 1)

        #clipped_mask = ~clipped_mask
        for enc_layer in self.enc_layers:
            x = enc_layer(x)

        x = x.transpose(0, 1)

        if self.lstm_layers is not None:
            #lengths = mask.sum(dim=1)
            #x_packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            for lstm in self.lstm_layers:
                #x_packed, _ = lstm(x_packed)
                x, _ = lstm(x)
            #x, _ = pad_packed_sequence(x_packed, batch_first=True)

        return x


class SleepFormer(nn.Module):
    def __init__(self, cfg):
        super(SleepFormer, self).__init__()
 
        self.CFG = cfg
        self.encoder = SleepformerEncoder(self.CFG)
        last_dim = self.CFG['sleepformer_dim'] * (2 if self.CFG['sleepformer_num_lstm_layers'] > 0 else 1)
        self.last_linear = nn.Linear(last_dim, 1)
        self.dropout = nn.Dropout(self.CFG['sleepformer_first_dropout'])
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x["X"]
        encoded_seq = self.encoder(x)
        encoded_seq = self.dropout(encoded_seq)

        output = self.last_linear(encoded_seq)
        #output = self.sigmoid(output)

        return output

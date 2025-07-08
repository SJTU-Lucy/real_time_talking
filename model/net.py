import torch
import torch.nn as nn
import math
from model.wav2vec2 import Wav2Vec2Model
from transformers import Wav2Vec2Config


def get_mask_from_lengths(lengths, max_len=None):
    lengths = lengths.to(torch.long)
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(lengths.shape[0], -1).to(lengths.device)
    mask = ids < lengths.unsqueeze(1).expand(-1, max_len)

    return mask


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class EncoderTransformer(nn.Module):
    def __init__(self, out_dim, d_model=512, nhead=8, num_layers=10, identity_dim=8, emo_dim=4):
        super().__init__()
        self._out_dim = out_dim

        self.audio_encoder_config = Wav2Vec2Config.from_pretrained(
            "C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h", local_files_only=True)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(
            "C:/Users/86134/Desktop/pretrain_weights/wav2vec2-base-960h", local_files_only=True)
        self.audio_encoder.feature_extractor._freeze_parameters()

        self.emotion = nn.Sequential(
            nn.Embedding(emo_dim, 256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, d_model)
        )

        self.identity = nn.Sequential(
            nn.Embedding(identity_dim, 256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        hidden_size = self.audio_encoder_config.hidden_size
        self.in_fn = nn.Linear(hidden_size, d_model)
        self.out_fn = nn.Linear(d_model, out_dim)

        self.pos_encoder = PositionalEncoding(d_model=d_model)

        for name, param in self.audio_encoder.named_parameters():
            param.requires_grad = False

        nn.init.constant_(self.out_fn.weight, 0)
        nn.init.constant_(self.out_fn.bias, 0)

    def forward(self, audio, seq_len, id, emo):
        id_embed = self.identity(id)
        emo_embed = self.emotion(emo)

        embeddings = self.audio_encoder(audio, seq_len=seq_len, output_hidden_states=True,
                                        attention_mask=None)

        hidden_states = embeddings.last_hidden_state
        layer_in = self.in_fn(hidden_states)
        layer_in = self.pos_encoder(layer_in)
        layer_in = layer_in + id_embed + emo_embed
        layer_in = self.encoder(layer_in)
        output = self.out_fn(layer_in)
        # output = torch.sigmoid(output)

        return output

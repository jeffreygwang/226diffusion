
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_w2v_embedding
import numpy as np
from sklearn import random_projection
import pickle

def one_param(m):
    """
    Get model first parameter
    """
    return next(iter(m.parameters()))

class EMA: # implements EMA for model stability 
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()
        self.channels = channels        
        self.mha = nn.MultiheadAttention(channels, 4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        size = x.shape[-1]
        x = x.view(-1, self.channels, size * size).swapaxes(1, 2)
        x_ln = self.ln(x)
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        attention_value = attention_value + x
        attention_value = self.ff_self(attention_value) + attention_value
        return attention_value.swapaxes(2, 1).view(-1, self.channels, size, size)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, time_dim=256, remove_deep_conv=False):
        super().__init__()
        self.time_dim = time_dim
        self.remove_deep_conv = remove_deep_conv
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128)
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256)


        if remove_deep_conv:
            self.bot1 = DoubleConv(256, 256)
            self.bot3 = DoubleConv(256, 256)
        else:
            self.bot1 = DoubleConv(256, 512)
            self.bot2 = DoubleConv(512, 512)
            self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=one_param(self).device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def unet_forwad(self, x, t):
        x1 = self.inc(x)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        if not self.remove_deep_conv:
            x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output
    
    def forward(self, x, t):
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.time_dim)
        return self.unet_forwad(x, t)


class UNet_conditional(UNet):
    def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, **kwargs):
        super().__init__(c_in, c_out, time_dim, **kwargs)

        self.w2v = load_w2v_embedding()
        self.index_to_dict = {
            0: "man",
            1: "woman", 
            2: "person",
            3: "nurse", 
            4: "philosopher"
        }

        # For the processing
        self.firsthalf = int(self.time_dim / 2)
        self.secondhalf = int(self.time_dim / 2)
        self.jls = {}

        self.precompute_JL()
    
    def precompute_JL(self):
        """
        Compute JL Projections, and save them. 
        Note that our guarantee is pretty shit lol (eps \approx 0.3)
        """
        keys = [0, 1, 2, 3, 4]
        print(f"JL projecting embeddings of {len(keys)} words...")

        transformer = random_projection.GaussianRandomProjection(self.secondhalf)
        words = [self.index_to_dict[i] for i in keys]
        embeddings = np.array([self.normalize_vector(self.w2v[word]) for word in words])
        self.embeddings = transformer.fit_transform(embeddings)

        # Save Them
        for i in range(len(keys)):
            self.jls[i] = self.embeddings[i]

        # Log some cosine similarity stats
        man_vs_person = cosine_similarity(self.embeddings[0], self.embeddings[2]) # man vs person 
        woman_vs_person = cosine_similarity(self.embeddings[1], self.embeddings[2]) # woman vs person 
        man_vs_nurse = cosine_similarity(self.embeddings[0], self.embeddings[3]) # man vs nurse 
        woman_vs_nurse = cosine_similarity(self.embeddings[1], self.embeddings[3]) # woman vs nurse 
        man_vs_phil = cosine_similarity(self.embeddings[0], self.embeddings[4]) # man vs phil 
        woman_vs_phil = cosine_similarity(self.embeddings[1], self.embeddings[4]) # woman vs phil 
        maxelems = [man_vs_person, woman_vs_person, man_vs_nurse, woman_vs_nurse, man_vs_phil, woman_vs_phil]
        print(f"Similarities: {maxelems}")

        # Save to disk
        with open('jls.pkl', 'wb') as file:
            pickle.dump(self.jls, file)
        
    def normalize_vector(self, v): # for embedding, make unit norm
        norm = np.linalg.norm(v)
        if norm == 0:
            return v  # Return the original vector if the norm is zero to avoid division by zero
        return v / norm

    def forward(self, x, t, y=None):
        """
        Assuming y is not none, this: 
        - Takes the time, sinusoidally encodes it to 128 dims. 
        - Then takes the words, grabs their JL projection, and concats.
        """
        t = t.unsqueeze(-1)
        t = self.pos_encoding(t, self.firsthalf)

        # Concatenate the w2vec embedding
        if y is not None:
            jl_embeds = np.array([self.jls[i.item()] for i in y])
            embed_tensor = torch.tensor(jl_embeds, dtype=torch.float32)
            embed_tensor = embed_tensor.to("cuda")
        else:
            embed_tensor = torch.zeros(t.shape[0], self.secondhalf)
            embed_tensor = embed_tensor.to("cuda")
        
        t = t.to("cuda")
        t = torch.cat((t, embed_tensor), dim=1)
        return self.unet_forwad(x, t)

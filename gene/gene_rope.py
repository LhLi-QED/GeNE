import math
import copy
import random
import numpy as np

import torch
import torch.nn as nn


class ParamInverseAngularVelocity(nn.Module):
    def __init__(self, config):
        super(ParamInverseAngularVelocity, self).__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.heads = config.num_attention_heads
        self.base = config.rope_theta
        self.pretrained_max_position_embeddings = config.max_position_embeddings
        self.critic_num_rotations = config.rope_scaling.get('num_rotations', 3)

        self.base_coef = nn.Parameter(torch.zeros(self.heads, self.dim // 2), requires_grad=True)
        self.specific_coef = nn.Parameter(torch.zeros(self.heads, self.dim // 2), requires_grad=True)

        self.alpha, self.beta = self._get_critical_dim()
        self._get_init_state()

    def _get_init_state(self):
        assert self.alpha < self.beta
        self.init_weight = ((torch.arange(self.dim // 2, dtype=torch.float32)
                             - self.alpha) / (self.beta - self.alpha)).clip(min=0., max=1.)
        # self.init_weight = (torch.arange(self.dim // 2, dtype=torch.float32)
        #                      - self.alpha) / (self.beta - self.alpha)  # ntk

    def _get_critical_dim(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        rot = inv_freq * self.pretrained_max_position_embeddings
        return 0, (rot < self.critic_num_rotations * 2 * torch.pi).nonzero()[0][0].item()

    def forward(self, r, inv_angvel):
        self.init_weight = self.init_weight.to(inv_angvel.device)
        ret = r ** self.init_weight * inv_angvel * torch.exp(self.base_coef)
        spe = self.specific_coef * r
        return ret + spe


class InverseAngularVelocity(nn.Module):
    def __init__(self, config):
        super(InverseAngularVelocity, self).__init__()
        self.dim = config.hidden_size // config.num_attention_heads
        self.heads = config.num_attention_heads
        self.base = config.rope_theta
        self.pretrained_max_position_embeddings = config.max_position_embeddings
        self.critic_num_rotations = config.rope_scaling.get('num_rotations', 3)
        self.alpha, self.beta = self._get_critical_dim()
        self._get_init_state()

    def _get_init_state(self):
        assert self.alpha < self.beta
        self.init_weight = ((torch.arange(self.dim // 2, dtype=torch.float32)
                             - self.alpha) / (self.beta - self.alpha)).clip(min=0., max=1.)

    def _get_critical_dim(self):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32) / self.dim))
        rot = inv_freq * self.pretrained_max_position_embeddings
        return 0, (rot < self.critic_num_rotations * 2 * torch.pi).nonzero()[0][0].item()

    def forward(self, r, inv_angvel):
        self.init_weight = self.init_weight.to(inv_angvel.device)
        ret = r ** self.init_weight * inv_angvel
        return ret


class RandomScale:
    def __init__(self, config):
        self._random_scale = 0
        self._random_position_ids = None
        self.max_position_embeddings = config.max_position_embeddings
        self.training_length = config.rope_scaling.get("training_seq_len", 16384)
        self.r = math.ceil(self.training_length / self.max_position_embeddings)

        self.rope_scaling_factor = config.rope_scaling.get('factor', 16.)
        self.s = self.rope_scaling_factor // self.r

        self._set_pdf()

    def _set_pdf(self):
        self.scales = np.arange(1, self.s + 1)
        p = np.array([self.pdf(i) for i in self.scales])
        z = p.sum()
        self.p = p / z

    def set_random(self, seq_len):
        self._random_scale = self.get_random_scale(seq_len)
        # self._random_scale = math.ceil(random.random() * self.random_scale_factor)
        # self._random_position_ids = self.get_random_position_ids(int(self._random_scale * self.training_length))

    def get_random_scale(self, seq_len):
        self.r = math.ceil(seq_len / self.max_position_embeddings)
        self.s = self.rope_scaling_factor // self.r
        self._set_pdf()
        sample = np.random.choice(self.scales, size=1, p=self.p)
        return sample[0]

    def get_random_position_ids(self, max_length):
        positions = torch.randperm(max_length - 2)[:self.training_length - 2].sort().values + 1
        positions = torch.concat(
            [torch.tensor([0]), positions, torch.tensor([max_length - 1])], dim=0
        )
        return positions

    def pdf(self, x):
        if self.s > 4:
            if x == 1 or x == self.s:
                return 0.25
            else:
                return 0.5 / (self.s - 2)
        else:
            return 1 / self.s

    @property
    def scale(self):
        return self._random_scale

    @property
    def position_ids(self):
        return self._random_position_ids


class GeneRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.base = config.rope_theta
        self.heads = config.num_attention_heads
        self.dim = config.hidden_size // config.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.training_seq_len = config.rope_scaling.get('training_seq_len', 16384)



        self.random_scale = config.rope_scaling.get('gene_random_scale', True)
        self.scale_const = self.training_seq_len / self.max_position_embeddings
        self.scale_factor = config.rope_scaling.get('factor', 16.) / self.scale_const
        assert self.scale_factor >= 1.0, "Scale factor must be greater than 1.0"

        if config.rope_scaling["type"] == 'gene':
            self.inv_angular_velocity = InverseAngularVelocity(config)
        elif config.rope_scaling["type"] == 'gene-param':
            self.inv_angular_velocity = ParamInverseAngularVelocity(config)
        else:
            raise ValueError("Unknown rope_scaling type: {}".format(config.rope_scaling["type"]))
        self._batch_random_scale = None

        self.max_seq_len_cached = 0

    def set_random_scale(self, brs: RandomScale):
        self._batch_random_scale = brs

    def _get_cos_sin(self, seq_len, device, dtype):
        inv_angvel = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim)

        if self.random_scale:
            if self.training:
                r = self._batch_random_scale.scale * math.ceil(seq_len / self.max_position_embeddings)
                t = torch.arange(seq_len, device=device, dtype=inv_angvel.dtype)
                # t = self._batch_random_scale.position_ids
                # t = t[:seq_len].to(device)
            else:
                r = max(self.scale_const * self.scale_factor, math.ceil(seq_len / self.max_position_embeddings))
                t = torch.arange(seq_len, device=device, dtype=inv_angvel.dtype)
        else:
            r = self.scale_factor * self.scale_const
            t = torch.arange(seq_len, device=device, dtype=inv_angvel.dtype)

        inv_freq = 1.0 / self.inv_angular_velocity(
            torch.tensor([r], device=device, dtype=torch.float32),
            inv_angvel
        )
        if len(inv_freq.shape) == 2:
            freqs = torch.einsum("t,hd->thd", t, inv_freq)
        else:
            freqs = torch.einsum("t,d->td", t, inv_freq)
            freqs = freqs[:, None, :]
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(dtype), emb.sin().to(dtype)

        return cos, sin

    def _set_cos_sin_cache(self, cos, sin):
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(self, seq_len, device, dtype):
        if not self.training:
            if self.max_seq_len_cached == 0:
                self.max_seq_len_cached = seq_len
                cos, sin = self._get_cos_sin(seq_len=seq_len, device=device, dtype=dtype)
                self._set_cos_sin_cache(cos, sin)
                return (
                    cos[:seq_len].to(dtype=dtype),
                    sin[:seq_len].to(dtype=dtype),
                )
            if seq_len > self.max_seq_len_cached:
                self.max_seq_len_cached = seq_len
                cos, sin = self._get_cos_sin(seq_len=seq_len, device=device, dtype=dtype)
                self._set_cos_sin_cache(cos, sin)
                return (
                    cos[:seq_len].to(dtype=dtype),
                    sin[:seq_len].to(dtype=dtype),
                )
            else:
                return (
                    self.cos_cached[:seq_len].to(dtype=dtype),
                    self.sin_cached[:seq_len].to(dtype=dtype),
                )

        cos, sin = self._get_cos_sin(seq_len=seq_len, device=device, dtype=dtype)
        return (
            cos[:seq_len].to(dtype=dtype),
            sin[:seq_len].to(dtype=dtype),
        )

    @torch.inference_mode
    def get_theta(self, s):
        device = self.inv_angular_velocity.base_coef.device
        inv_angvel = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float32).to(device) / self.dim)
        return (1.0 / self.inv_angular_velocity(torch.tensor([s], device=device, dtype=torch.float32),
                                                inv_angvel)).cpu()

    def get_param(self):
        return [self.inv_angular_velocity.base_coef, self.inv_angular_velocity.specific_coef]


import torch
import torch.nn as nn
from torch.nn import Transformer


# YOLO V3 model:

# Detection with transformers:
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2


        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )


    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)

        return x

    class Attention(nn.Module):
        def __init__(self, dim, n_heads=12, qkv_bias=True, attn_drop=0., proj_drop=0.):
            super().__init__()
            assert dim % n_heads == 0, 'dim should be divisible by num_heads'
            self.n_heads = n_heads
            self.head_dim = dim // n_heads
            self.scale = self.head_dim ** -0.5

            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
            self.attn_drop = nn.Dropout(attn_drop)
            self.proj = nn.Linear(dim, dim)
            self.proj_drop = nn.Dropout(proj_drop)

        def forward(self, x):
            n_samples, n_tokens, dim = x.shape
            assert (dim == self.dim), "Dim != self.dim"

            qkv = self.qkv(x).reshape(n_samples, n_tokens, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
            # q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)

            # x = (attn @ v).transpose(1, 2).reshape(n_samples, n_tokens, dim)
            x = (attn @ v).transpose(1, 2).flatten(2)
            x = self.proj(x)
            x = self.proj_drop(x)
            return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Block(nn.Module):
    def __init__(self):




'''''
class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (self.head_dim * heads == embed_size), "Embed size needs to be div by heads"
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)



    def forward(self, values, keys, query, mask):
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)


        energy = torch.einsum()

'''









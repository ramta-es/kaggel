import torch
import torch.nn as nn
from torch.nn import Transformer


# YOLO V3 model:

# Vision Transformer:
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
        x = (attn @ v).transpose(1, 2).flatten(2) # weighted average
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
    """Transformer Block
    Parameters
    ------------
    dim: int
    Embedding dimension

    n_heads: int
    Number of attention heads

    mlp_ration: float
    Determines the hidden dimension size of the MLP module with respect to dim

    qkv_bias: bool
    If true then we include bias to the query, key and value projections.

    p, attn_p: float
    Dropout probability

    Attributes
    ------------
    norm1, norm2: LayerNorm
    Layer normalization

    attn: Attention
    Attention module

    mlp: MLP
    MLP module


    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, n_heads=n_heads, qkv_bias=qkv_bias, attn_drop=attn_p, proj_drop=p)
        self.norm2 = nn.LayerNorm(dim, 1e-6)
        hidden_featurs = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_featurs, out_features=dim)

    def forwared(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """
    Parameters
    ------------
    img_size: int
    height and width of the image(squared image)

    patch_size: int
    height and width of the patch(squared patch)

    in_channels: int
    number of input channels

    n_classes
    numbr of classes

    embed_dim: int
    Dimensionality of the token/ patch embeddings

    depth: int
    number of blocks

    n_heads: int
    number of attention heads

    mlp_ration: float
    Determines the hidden dimension of the MLP module

    qkv_bias: bool
    if true then we include the bias to the query, key and value

    p, attn_p: float
    Dropout probability

    Attributes
    ------------
    patch_embed: PatchEmbed
    instance of PatchEmbed layer

    cls_token: nn.Parameter
    learnable parameter that will represent the first token in sequence
    it has the embed_dim element

    pos_embed: nn.Parameter
    positional embedding of the cls token + all the patches
    it has (n_patches + 1) * embed_dim elenemts

    pos_drop: nn.Dropout
    Dropout layer

    blocks: nn.ModuleList
    list of block modules

    norm: nn.LayerNorm
    layer normalization
    """
    def __init__(self, img_size=384, patch_size=16, in_chans=3, n_classes=1000,
                 embed_dim=768, depth=12, n_heads=12, mlp_ratio=4., qkv_bias=True,
                 p=0., attn_p=0.):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                     in_chans=in_chans, embed_dim=embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.path_embed.n_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList([Block(dim=embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio,
                                           qkv_bias=qkv_bias, p=p, attn_p=attn_p)
                                     for _ in range(depth)
                                     ]
                                    )
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        n_samples = x.shape[0]
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x+self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0] #just the cls token
        x = self.head(cls_token_final)

        return x










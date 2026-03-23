from dataclasses import dataclass

import time
import math
import inspect
import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F


@dataclass
class Config:
    sequence_length: int = 1024
    vocab_size: int = 50257
    num_layers: int = 12
    num_heads: int = 12
    d_model: int = 768


# ------------ DATALOADER ------------------------


class MartinDatalaoder:
    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open("data/text-data/input.txt") as f:
            text_data = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text_data)
         # tokens  =  [ ]
        self.tokens = torch.tensor(tokens)
        
        print(f"loaded total tokens are ==== {len(self.tokens)}")
        print(f"1 epoch = {len(self.tokens) // B * T} batches of size {B} and sequence length {T}")

        self.cp = 0

        # [ 23, 45, 67, 89, 12, 34, 56, 78, 90, 123, 456, 789, 101112, 131415 ]


    def next_batch(self):

        B, T = self.B, self.T # 4 * 8 = 32

        buf = self.tokens[self.cp : self.cp + B * T + 1] # 33 

        x = buf[:-1].view(B, T) # 0 - 31 4 x 8 
        y = buf[1:].view(B, T) # 1 - 32

        # print hte example of 4 * 8 batch 
#         Row 0: [[ 23,  45,  67,  89,  12,  34,  56,  78]    ← tokens 0-7 -> 4 * 8 * 768
        # Row 1: [ 90, 123, 456, 789,  11,  14,  55,  77]    ← tokens 8-15
        # Row 2: [ 88,  99,  21,  33,  44,  66, 100, 200]    ← tokens 16-23
        # Row 3: [300, 400, 500, 600, 700, 800, 900,1000]]    ← tokens 24-31


        # Row 0: [ 45,  67,  89,  12,  34,  56,  78,  90]    ← tokens 1-8
        # Row 1: [123, 456, 789,  11,  14,  55,  77,  88]    ← tokens 9-16
        # Row 2: [ 99,  21,  33,  44,  66, 100, 200, 300]    ← tokens 17-24
        # Row 3: [400, 500, 600, 700, 800, 900,1000,1100]    ← tokens 25-32

        self.cp += B * T

        if self.cp + B * T + 1 > len(self.tokens):
            self.cp = 0

        return x, y


# ----  GPT transformer
#


class CasualAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.d_model % config.num_heads == 0

        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model)
        self.c_proj = nn.Linear(config.d_model, config.d_model)
        # self.c_proj.NANOGPT_SCALE_INIT = 1

        # regularization
        self.num_heads = config.num_heads
        self.d_model = config.d_model
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.sequence_length, config.sequence_length)).view(
                1, 1, config.sequence_length, config.sequence_length
            ),
        )

    def forward(self, x):

        B, T, C = x.size()  # batch_size, sequence_length, embedding_dimensionality ( d_model )

        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)

        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)  # (B, nh, T, hs)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.c_proj(y)

        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.d_model, 4 * config.d_model)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.d_model, config.d_model)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.d_model)
        self.attn = CasualAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    

##  THIS IS PEN 
#   768 768 738 [ ]

#   768 768 768


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config 

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.d_model),  # word_token_embeddings
                wpe=nn.Embedding(config.sequence_length, config.d_model),
                h=nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)]),
                ln_f=nn.LayerNorm(config.d_model),
            )

          
        )

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # wieght sharing trick
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.num_layers) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.000, std=std)

            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        B, T = idx.size()

        assert T <= self.config.sequence_length, f"Cannot forward sequence of lenght {T}, sequence lenght is only {self.config.sequence_length}"

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)

        x = tok_emb + pos_emb

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        logits = self.lm_head(x)

        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


# ------------------------------ Training Loop -------------------------------


device = "cpu"

# if torch.cuda.is_available():
#     device = "cuda"
# elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
#     device = "mps"

print(f"USING DEVICE: {device}")

torch.manual_seed(333)

if torch.cuda.is_available():
    torch.cuda.manual_seed(333)


B = 16
T = 1024

data_loader = MartinDatalaoder(B=16, T=1024)


model = GPT(Config(vocab_size=50304))
model.to(device)
# model torch.compile

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-6)#@ 3 * 10-2 3 * 10 - 3
### Learning rate Scheduler


total_t0 = time.time()

for i in range(500):
    t0 = time.time()

    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)

    optimizer.zero_grad()
    logits, loss = model(x, y)
    loss.backward()

    optimizer.step()

    t1 = time.time()
    dt = t1 - t0
    token_per_sec = (B * T) / dt
    print(f"setp {i} |  loss: {loss.item():.4f} | tok/sec: {token_per_sec:.0f} | dt: {dt * 1000:.2f}ms")

total_td = time.time() - total_t0
print(f"\nTotal Time: {total_td:.2f}s")

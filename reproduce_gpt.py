from dataclasses import dataclass 

import math
import inspect
import torch 
import torch.nn as nn
from torch.nn  import functional as F


@dataclass 
class GPTConfig:
    block_size: int = 1024 #max sequence length
    vocab_size: int = 50257 #vocabulary size, e.g. GPT-2 has 50257 tokens
    n_layer: int = 12 # number of layers 
    n_head: int = 12 # number of heads
    n_embd: int = 768 # embedding 


class CasualSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)

        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.n_head = config.n_head
        self.n_embd = config.n_embd 

        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    
    def forward(self,x ):

        B,T,C = x.size() 
        """
        Batch Size, Sequence Length, Embedding Dimensionality ( n_embd)
        nh is "Number of heads", hs is "head Size", and C "Number of channels" = nh * hs
        e.g. in GPT-2 ( 124M ) , n_head=12, hs=64, so nh*hs=C=768 channels in transformer
        """

        qkv = self.c_attn(x) # (B,T,3*C)

        q, k, v = qkv.split(self.n_embd, dim=2) # (B,T,C)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh, T, hs)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B,nh, T, hs)

        # att = ( q @ k.transpose(-2, -1)) * ( 1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        # att = F.softmax(att,dim=-1)
        # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        #Using the Flash Attention
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True)

        y = y.transpose(1,2).contiguous().view(B,T,C) # reassemble all head outputs side by side
        #output projection 
        y = self.c_proj(y) # output projection 

        return y

class MLP(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    
    def __init__(self,config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CasualSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x




class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # word token embedding
            wpe = nn.Embedding(config.block_size, config.n_embd), # position embedding
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # transformer
            ln_f = nn.LayerNorm(config.n_embd)
        ))


        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)


        # weight sharing scheme 
        self.transformer.wte.weight = self.lm_head.weight # share the weights of the token embedding with the output layer

        # initialize params weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # initialize linear layers with normal distribution
            std = 0.02
            if hasattr(module, 'NANOGPT_SCALE_INIT'):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            # initialize word embeddings 
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def configure_optimizers(self, weight_decay, learning_rate, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9,0.95), **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def forward(self, idx, targets=None):

        B, T = idx.size() # 4,32

        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"

        pos = torch.arange(0,T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos) # positional embd of shape (T, n_embd( C = 768)) = (32, 768)
        tok_emb = self.transformer.wte(idx) # token embd of shape ( B, T, n_embd) = (4,32,768)

        # here automatic conversion happens 
        x = tok_emb + pos_emb # (B, T, n_embd) # (4, 32, 768) input token indices + positional embedding

        for block in self.transformer.h:
            x = block(x)
        
        x = self.transformer.ln_f(x) # each x is ( B, T, C) = ( 4, 32, 768)
        logits = self.lm_head(x) # (B, T, V) = (4, 32, 50257) logits for each token in the vocabulary

        # logits.view(-1, logits.size(-1)) ( B*T, V) = (128, 50257)
        # y.view(-1) ( B*T) = (128,)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @classmethod
    def from_pretrained(cls, model_type):
       
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]

        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints


        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [ k for k in sd_keys if not k.endswith('.attn.bias')]

        #HUFFINGFACE TRANSFORMER
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model


# ----------------------------------------------------------

import tiktoken 

class DataLoaderLite:

    def __init__(self, B, T):
        self.B = B
        self.T = T

        with open('input.txt', 'r') as f:
            text = f.read()

        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)

        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens from input.txt")
        print(f" 1 epoch = {len(self.tokens) // (B * T)} batches of size {B} and sequence length {T}")


        self.current_position = 0


    def next_batch(self):

        B,T = self.B, self.T
        
        buf = self.tokens[self.current_position:self.current_position + B * T + 1] # buffer of token indices

        x = (buf[:-1]).view(B,T) # (B, T) = (4, 32) input token indices
        y = (buf[1:]).view(B,T) 

        self.current_position += B * T

        if self.current_position + ( B * T + 1)  > len(self.tokens):
            self.current_position = 0 # reset to the beginning of the file
        
        return x,y


# ------------------------------------------------------------

import time 

device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = "mps"

print(f"Using device: {device}")

torch.manual_seed(3333)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3333)

# data_loader = DataLoaderLite(B=4, T=32) # batch size 4, sequence length 32
data_loader = DataLoaderLite(B=16, T=1024) 

torch.set_float32_matmul_precision('high')

#get logits 
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model)



#defining learning rate 
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 10
max_steps = 50
def get_lr(it):
    # linear warmup 
    if it < warmup_steps:
        return max_lr * (it+1) / warmup_steps
    # it> lr decays return min lr  
    if it > max_steps:
        return min_lr
    # if it is inbetween decay down to min lr 
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <=1 
    coeff = 0.5 * ( 1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

#optimizer 
optimizer = model.configure_optimizers(weight_decay=0.1, learning_rate=3e-4, device_type=device)

for step in range(50):

    t0 = time.time()
    x, y = data_loader.next_batch()
    x, y = x.to(device), y.to(device)
    optimizer.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16):
        logits, loss = model(x, y)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping

    lr = get_lr(step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()

    torch.cuda.synchronize() #wait for the gpu operation 
    t1 = time.time()
    dt = ( t1 - t0 ) * 1000
    tokens_per_sec = (data_loader.B * data_loader.T) / (t1 - t0)
    print(f"step {step:4d} | loss: {loss.item():.6f} | lr {lr:.4e} | norm: {norm:.4f} |  dt: {dt:.2f}ms | tok/sec: {tokens_per_sec:.2f}")


import sys; sys.exit(0)




#-------------------------------------------

num_return_seq = 5
max_length = 30


# model = GPT.from_pretrained('gpt2')
model = GPT(GPTConfig())
model.eval()
model.to(device)


# tokens = enc.encode("Hello, I'm a language model, ")
# tokens = torch.tensor(tokens, dtype=torch.long)
# tokens = tokens.unsqueeze(0).repeat(num_return_seq,1) # (5,8)

x = tokens.to('cuda')

#generate! right now x is (B, T) where B = 5, T = 8
# set the seed to 42 
torch.manual_seed(42)
torch.cuda.manual_seed(42)

while x.size(1) < max_length:

    with torch.no_grad():
        logits = model(x)
        logits = logits[:,-1,:] # only last token
        probs = F.softmax(logits, dim=-1)

        topk_probs, topk_indices = torch.topk(probs, 50, dim=-1) 

        ix = torch.multinomial(topk_probs, 1) # sample from the topk probs

        xcol = torch.gather(topk_indices, -1, ix)
        x = torch.cat((x, xcol), dim=1)

for i in range(num_return_seq):
    tokens = x[i, :max_length].tolist()
    decoded = enc.decode(tokens)
    print(">", decoded)

print("-"* 10, "working good")

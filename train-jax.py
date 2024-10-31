import os
import time
from enums import InitFrom
from model import FQConfig, FrostQuantix
from utils import download_file, estimate_loss, get_batch, get_lr
from flax.training import checkpoints
import jax 
import jax.numpy as jnp
import numpy as np
import flax.linen as nn


### Static Variables

## I/O
out_dir = 'out'

## interval
eval_interval = 100
log_interval = 1

## Evaluation
eval_inters = 300
eval_only = False

## data
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch size
batch_size = 1   # if gradient_accumulation_steps > 1, this is the micro batch size
block_size = 128 # 1024

## Model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = False

## Adamw Optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000   # total number of iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, 0 to disable

# Learning rate
decay_lr = True
warmup_iters = 2000
lr_decay_iters = 600000
min_lr = 6e-5

## ETC
init_from = InitFrom.SCRATCH
always_save_checkpoint = False
compile = False 
seed_offset = 0
seed = jax.random.PRNGKey(1337 * seed_offset)
device = jax.devices()[0]

## Checkpoint
iter_num = 0
best_val_loss = 1e9
meta_vocab_size = 50304
model_args = dict(n_layer = n_layer, n_head = n_head, n_embd = n_embd, dropout = dropout, bias = bias, vocab_size=None)
local_iter_num = 0
running_mfu = -1.0

### End Static Variables

### Config
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
###

print("Using HuggingFace binaries...")
download_file('https://huggingface.co/datasets/VatsaDev/TinyText/resolve/main/valtotal1.bin','valtotal.bin')
download_file('https://huggingface.co/datasets/VatsaDev/TinyText/resolve/main/traintotal1.bin', 'traintotal.bin')

train_data = np.memmap('traintotal.bin', dtype=np.uint16, mode='r')
val_data = np.memmap('valtotal.bin', dtype=np.uint16, mode='r')
print("got HuggingFace binaries, batching")

model: FrostQuantix

if init_from == InitFrom.SCRATCH:
    print("Initializing model from scratch")

    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = FQConfig(**model_args)
    model = FrostQuantix(gptconf, 768, 3, 768)
elif init_from == InitFrom.RESUME:
    print(f"Resuming model training from {out_dir}")
    checkpoint_path = os.path.join(out_dir, 'checkpoint.pt')
    checkpoint = checkpoints.restore_checkpoint(checkpoint_path, target=None)
    checkpoint_model_args = checkpoint['model_args']

    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]

    gptconfig = FQConfig(**model_args)
    model = FrostQuantix(gptconfig, 768, 3, 768)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'

    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size
model.to(device)

optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=learning_rate, betas=(beta1, beta2))
if init_from == InitFrom.RESUME:
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free memory

print("Start training")
x, y = get_batch('train', train_data, val_data, block_size, batch_size, device)
t0 = time.time()

while True:
    print(f"Iteration {iter_num}")
    lr = get_lr(iter_num, warmup_iters, learning_rate, lr_decay_iters, min_lr) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    if iter_num % eval_interval == 0:
        losses = estimate_loss()
        print(f"Step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    model: model.state_dict(),
                    optimizer: optimizer.state_dict(),
                    model_args: model_args,
                    iter_num: iter_num,
                    best_val_loss: best_val_loss,
                    config: config
                }
                print(f"Saving checkpoint to {out_dir}")
                jnp.save(checkpoint, os.path.join(out_dir, 'checkpoint.pt'))
    
    if iter_num == 0 and eval_only:
        break

    for micro_step in range(gradient_accumulation_steps):
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps

        x, y = get_batch('train', train_data, val_data, block_size, batch_size, device)
    
    optimizer.zero_grad(set_to_none = True)

    mfu = -1.0
    t1 = time.time()
    dt = t1 - t0
    t0 = t1

    if iter_num % log_interval == 0:
        lossf = loss.item() * gradient_accumulation_steps
        
        if local_iter_num >= 5:
            mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
    

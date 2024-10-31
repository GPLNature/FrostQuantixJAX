import math
import requests
import jax
import jax.numpy as jnp

def download_file(url, file_name):
  response = requests.get(url, stream=True)
  if response.status_code == 200:
    with open(file_name, 'wb') as f:
      for chunk in response.iter_content(chunk_size=104857600):
        if chunk:
          f.write(chunk)
    print(f"got: {file_name}")
  else:
    print('Error downloading file:', response.status_code)

def get_batch(split, train_data, val_data, block_size, batch_size, device):
    data = train_data if split == 'train' else val_data

    ix = jax.random.randint(len(data) - block_size, (batch_size,))
    x = jnp.stack([jnp.array(data[i:i + block_size], dtype=jnp.int64) for i in ix])
    y = jnp.stack([jnp.array(data[i + 1:i + 1 + block_size], dtype=jnp.int64) for i in ix])

    jax.device_put(x, device=device)
    jax.device_put(y, device=device)

@jax.jit
def estimate_loss(model, get_batch, eval_iters):
    out = {}

    for split in ['train', 'val']:
        losses = jnp.zeros(eval_iters)  # Initialize losses with JAX arrays
        for k in range(eval_iters):
            X, Y = get_batch(split)
            # Compute loss without gradients
            _, loss = model(X, Y)  # Ensure model returns (logits, loss)
            if loss is not None:
                losses = losses.at[k].set(loss)  # Update losses at index k
            else:
                losses = losses.at[k].set(-1)  # If loss is None

        out[split] = jnp.mean(losses)  # Compute the mean loss for the split

    return out

@jax.jit
def forward(model, params, inputs):
    return model.apply(params, inputs)

def get_lr(it, warmup_iters, learning_rate, lr_decay_iters, min_lr):
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (learning_rate - min_lr)

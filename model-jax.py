from dataclasses import dataclass
from jax.nn import gelu
import jax.random as jrd
import jax
import flax.linen as nn
import jax.numpy as jnp


@dataclass
class FQConfigJAX:
    block_size: int = 1024
    vocab_size: int = (
        50304  # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    )
    group_size: int = 1
    window_size: int = 1
    n_layer: int = 12
    n_head: int = 12
    # in_features
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = (
        True  # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    )


class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False"""

    use_bias: bool
    epsilon: float = 1e-5

    @nn.compact
    def __call__(self, input):
        # Compute the mean and variance
        mean = jnp.mean(input, axis=-1, keepdims=True)
        var = jnp.var(input, axis=-1, keepdims=True)

        # Normalize the input
        normalized = (input - mean) / jnp.sqrt(var + self.epsilon)

        # Scale and shift it
        weight = self.param('weight', nn.initializers.ones, input.shape[-1])
        bias = self.param('bias', nn.initializers.zeros, input.shape[-1]) if self.use_bias else None

        return normalized * weight + bias


class RMSNorm(nn.Module):

    dim: int
    epsilon: float = 1e-8

    @nn.compact
    def __call__(self, input):
        scale = self.dim **- 0.5
        g = self.param('g', nn.initializers.ones, input.shape[-1])

        norm = jnp.linalg.norm(input, axis=-1, keepdims=True) * scale
        return input / (norm + self.epsilon) * g


class CausalSelfAttention(nn.Module):

    config: FQConfigJAX

    # during training deterministic parameter should be False
    # if evaluation phase deterministic parameter should be True
    @nn.compact
    def __call__(self, input, deterministic: bool = False):
        assert self.config.n_embd % self.config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        b, t, u = input.shape
        att_dropout = nn.Dropout(self.config.dropout)
        resid_dropout = nn.Dropout(self.config.dropout)

        block_array = jnp.ones((self.config.block_size, self.config.block_size))
        bias = jnp.tril(block_array).reshape(1, 1, self.config.block_size, self.config.block_size)

        c_attn = nn.Dense(3 * self.config.n_embd, use_bias=self.config.bias)(input)
        c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.bias)

        q, k, v = c_attn(input).split(3, axis=-1)

        q = q.reshape(b, t, self.config.n_head, u // self.config.n_head).transpose(0, 2, 1, 3)
        k = k.reshape(b, t, self.config.n_head, u // self.config.n_head).transpose(0, 2, 1, 3)
        v = v.reshape(b, t, self.config.n_head, u // self.config.n_head).transpose(0, 2, 1, 3)

        # attention = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(k.shape[-1]))
        attention = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (1.0 / jnp.sqrt(k.shape[-1]))
        attention += jnp.where(bias == 0, -jnp.inf, 0.0)
        attention = nn.softmax(attention, axis=-1)
        attention = att_dropout(attention, deterministic=deterministic)

        y = jnp.matmul(attention, v)
        y = y.transpose(0, 2, 1, 3).reshape(b, t, u) # doesn't need contiguous() in JAX

        # output projection
        return resid_dropout(c_proj(y), deterministic=deterministic)


class FrostQuantixAttentionJAX(nn.Module):

    config: FQConfigJAX
    head_dim: float

    def _split_heads(self, x: jnp.ndarray):
        return x.reshape(x[0], x[1], self.config.n_head, self.head_dim).transpose(0, 2, 1, 3)

    def _rel_shift(self, x: jnp.ndarray):
        zero_pad = jnp.zeros((x[0], 1, *x.shape[2:]), dtype=x.dtype)
        return jnp.concatenate([zero_pad, x], axis=1)[:, :-1]

    def _merge_heads(self, x: jnp.ndarray):
        return x.swapaxes(1, 2).reshape(x[0], x[2], self.config.n_embd)

    # during training deterministic parameter should be False
    # if evaluation phase deterministic parameter should be True
    @nn.compact
    def __call__(self, input, deterministic: bool = False):
        assert self.config.n_embd % self.config.n_head == 0, "Embedding dimension must be divisible by number of heads."
        b, t, c = input.shape

        self.head_dim = head_dim = self.config.n_embd // self.config.n_head
        scale = head_dim ** -0.5
        # max_seq_len = self.config.block_size
        rel_pos_enc = nn.Embed(2 * self.config.block_size - 1, head_dim)

        attn_dropout = nn.Dropout(self.config.dropout)
        resid_dropout = nn.Dropout(self.config.dropout)

        c_attn = nn.Dense(3 * self.config.n_embd, use_bias=self.config.bias)(input)
        c_proj = nn.Dense(self.config.n_embd, use_bias=self.config.bias)

        # Split and reshape for multi-head attention
        q, k, v = c_attn(input).split(3, axis=-1)
        q, k, v = map(self._split_heads, (q, k, v))

        # Compute attention scores with relative positional encoding
        q = q * scale
        attn_scores = jnp.matmul(q, k.transpose(0, 1, 3, 2))
        rel_pos_enc = rel_pos_enc(jnp.arange(t))
        rel_pos_enc = self._rel_shift(rel_pos_enc)
        attn_scores += jnp.einsum("bhid,jd->bhij", q, rel_pos_enc)

        # Apply causal mask
        t_array = jnp.ones((t, t))
        causal_mask = jnp.tril(t_array).astype(jnp.bool).reshape(1, 1, t, t)
        attn_scores = jnp.where(~causal_mask, -jnp.inf, attn_scores)

        # Compute attention probabilities and values
        attn_probs = nn.softmax(attn_scores, axis=-1)
        attn_probs = attn_dropout(attn_probs, deterministic=deterministic)
        y = jnp.matmul(attn_probs, v)

        # Re-assemble all head outputs side by side
        y = self._merge_heads(y)

        # Output projection
        y = resid_dropout(c_proj(y))
        return y


class MLP(nn.Module):

    config: FQConfigJAX

    # during training deterministic parameter should be False
    # if evaluation phase deterministic parameter should be True
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        x = nn.Dense(4 * self.config.n_embd, use_bias=self.config.bias)(x)
        x = gelu(x)
        x = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        x = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
        return x


class GatedMLP(nn.Module):

    # during training deterministic parameter should be False
    # if evaluation phase deterministic parameter should be True
    @nn.compact
    def __call__(self, x, deterministic: bool = False):
        x = nn.Dense(4 * self.config.n_embd, use_bias=self.config.bias)(x)
        x = gelu(x)
        x = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        gate = nn.Dense(self.config.n_embd, use_bias=self.config.bias)(x)
        gate = nn.sigmoid(gate)
        x = x * gate
        x = nn.Dropout(self.config.dropout)(x, deterministic=deterministic)
        return x


class FrostQuantixBlock(nn.Module):

    config: FQConfigJAX

    @nn.compact
    def __call__(self, x):
        rms_norm1 = RMSNorm(self.config.n_embd)(x)
        x = x + CausalSelfAttention(self.config)(rms_norm1)
        rms_norm2 = RMSNorm(self.config.n_embd)(x)
        x = x + GatedMLP(self.config)(rms_norm2)
        return x


class FrostQuantixJAX(nn.Module):

    config: FQConfigJAX

    def _softmax_cross_entropy(self, logits, labels):
        log_probs = nn.log_sigmoid(logits)
        loss = -jnp.sum(log_probs * nn.one_hot(labels, num_classes=logits.shape[-1]), axis=-1)
        return loss.mean()

    @nn.compact
    def __call__(self, idx, targets=None):
        b, t = idx.shape
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = jnp.arange(t, dtype=jnp.int64)

        # Define Layer
        wte = self.param('wte', nn.initializers.xavier_uniform(), (self.config.vocab_size, self.config.n_embd))
        wpe = self.param('wpe', nn.initializers.xavier_uniform(), (self.config.block_size, self.config.n_embd))

        tok_emb = idx @ wte
        pos_emb = wpe[pos]

        x = tok_emb + pos_emb
        x = nn.Dropout(self.config.dropout)(x, deterministic=False)

        for _ in range(self.config.n_layer):
            x = FrostQuantixBlock(self.config)(x)

        x = LayerNorm(epsilon=self.config.n_embd, use_bias=self.config.bias)

        lm_head = nn.Dense(self.config.vocab_size, use_bias=False)

        logits = lm_head(x)
        if targets is not None:
            loss = self._softmax_cross_entropy(logits, targets)
        else:
            logits = lm_head(x[:, -1, :])
            loss = None

        return logits, loss

    @jax.jit
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.block_size else idx[:, -self.config.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = jax.lax.top_k(logits, min(top_k, logits.shape[-1]))
                logits[logits < v[:, [-1]]] = -jnp.inf
            
            probs = nn.softmax(logits, axis=-1)
            idx_next = jrd.choice(jrd.PRNGKey(0), logits.shape[-1], shape=(1,), p=probs)
            idx = jnp.concatenate([idx, idx_next], axis=1)

        return idx

    # Include other necessary methods like crop_block_size, from_pretrained, etc.

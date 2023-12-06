import math
from sentencepiece import SentencePieceProcessor

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_unflatten
import llm

from typing import Optional


@llm.hookimpl
def register_models(register):
    register(MlxLlama())


class MlxLlama(llm.Model):
    model_id = "mlx-llama"

    def __init__(self):
        self._models = {}
        self._tokenizers = {}

    class Options(llm.Options):
        model: str
        tokenizer: str
        num_tokens: Optional[int] = 200
        temperature: Optional[float] = 0.8

    def execute(self, prompt, stream, response, conversation):
        tokenizer = SentencePieceProcessor(model_file=prompt.options.tokenizer)
        model = load_model(prompt.options.model)
        return generate(
            tokenizer,
            model,
            prompt.prompt,
            prompt.options.temperature,
            prompt.options.num_tokens,
        )


class LlamaAttention(nn.Module):
    def __init__(self, dims: int, num_heads: int):
        super().__init__()

        self.num_heads = num_heads

        self.rope = nn.RoPE(dims // num_heads, traditional=True)
        self.query_proj = nn.Linear(dims, dims, bias=False)
        self.key_proj = nn.Linear(dims, dims, bias=False)
        self.value_proj = nn.Linear(dims, dims, bias=False)
        self.out_proj = nn.Linear(dims, dims, bias=False)

    def __call__(self, queries, keys, values, mask=None, cache=None):
        queries = self.query_proj(queries)
        keys = self.key_proj(keys)
        values = self.value_proj(values)

        # Extract some shapes
        num_heads = self.num_heads
        B, L, D = queries.shape

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, num_heads, -1).transpose(0, 2, 1, 3)

        # Add RoPE to the queries and keys and combine them with the cache
        if cache is not None:
            key_cache, value_cache = cache
            queries = self.rope(queries, offset=key_cache.shape[2])
            keys = self.rope(keys, offset=key_cache.shape[2])
            keys = mx.concatenate([key_cache, keys], axis=2)
            values = mx.concatenate([value_cache, values], axis=2)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        # Finally perform the attention computation
        scale = math.sqrt(1 / queries.shape[-1])
        scores = (queries * scale) @ keys.transpose(0, 1, 3, 2)
        if mask is not None:
            scores = scores + mask
        scores = mx.softmax(scores, axis=-1)
        values_hat = (scores @ values).transpose(0, 2, 1, 3).reshape(B, L, -1)

        # Note that we return the keys and values to possibly be used as a cache
        return self.out_proj(values_hat), (keys, values)


class LlamaEncoderLayer(nn.Module):
    def __init__(self, dims: int, mlp_dims: int, num_heads: int):
        super().__init__()

        self.attention = LlamaAttention(dims, num_heads)

        self.norm1 = nn.RMSNorm(dims)
        self.norm2 = nn.RMSNorm(dims)

        self.linear1 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear2 = nn.Linear(dims, mlp_dims, bias=False)
        self.linear3 = nn.Linear(mlp_dims, dims, bias=False)

    def __call__(self, x, mask=None, cache=None):
        y = self.norm1(x)
        y, cache = self.attention(y, y, y, mask, cache)
        x = x + y

        y = self.norm2(x)
        a = self.linear1(y)
        b = self.linear2(y)
        y = a * mx.sigmoid(a) * b
        y = self.linear3(y)
        x = x + y

        return x, cache


class Llama(nn.Module):
    def __init__(
        self, num_layers: int, vocab_size: int, dims: int, mlp_dims: int, num_heads: int
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, dims)
        self.layers = [
            LlamaEncoderLayer(dims, mlp_dims, num_heads) for _ in range(num_layers)
        ]
        self.norm = nn.RMSNorm(dims)
        self.out_proj = nn.Linear(dims, vocab_size, bias=False)

    def __call__(self, x):
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        x = self.embedding(x)
        for l in self.layers:
            x, _ = l(x, mask)
        x = self.norm(x)
        return self.out_proj(x)

    def generate(self, x, temp=1.0):
        cache = []

        # Make an additive causal mask. We will need that to process the prompt.
        mask = nn.MultiHeadAttention.create_additive_causal_mask(x.shape[1])
        mask = mask.astype(self.embedding.weight.dtype)

        # First we process the prompt x the same was as in __call__ but
        # save the caches in cache
        x = self.embedding(x)
        for l in self.layers:
            x, c = l(x, mask=mask)
            # We store the per layer cache in a simple python list
            cache.append(c)
        x = self.norm(x)
        # We only care about the last logits that generate the next token
        y = self.out_proj(x[:, -1])
        y = mx.random.categorical(y * (1 / temp))

        # y now has size [1]
        # Since MLX is lazily evaluated nothing is computed yet.
        # Calling y.item() would force the computation to happen at
        # this point but we can also choose not to do that and let the
        # user choose when to start the computation.
        yield y

        # Now we parsed the prompt and generated the first token we
        # need to feed it back into the model and loop to generate the
        # rest.
        while True:
            # Unsqueezing the last dimension to add a sequence length
            # dimension of 1
            x = y[:, None]

            x = self.embedding(x)
            for i in range(len(cache)):
                # We are overwriting the arrays in the cache list. When
                # the computation will happen, MLX will be discarding the
                # old cache the moment it is not needed anymore.
                x, cache[i] = self.layers[i](x, mask=None, cache=cache[i])
            x = self.norm(x)
            y = self.out_proj(x[:, -1])
            y = mx.random.categorical(y * (1 / temp))

            yield y


def generate(tokenizer, model, prompt, temperature, num_tokens, write_every=1):
    x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(prompt)])
    skip = 0
    tokens = []
    for token in model.generate(x, temperature):
        tokens.append(token)

        if len(tokens) == 1:
            # Actually perform the computation to measure the prompt processing time
            mx.eval(token)

        if len(tokens) >= num_tokens:
            break

        elif (len(tokens) % write_every) == 0:
            # It is perfectly ok to eval things we have already eval-ed.
            mx.eval(tokens)
            s = tokenizer.decode([t.item() for t in tokens])
            yield s[skip:]
            skip = len(s)

    mx.eval(tokens)
    s = tokenizer.decode([t.item() for t in tokens])
    yield s[skip:]


def few_shot_generate(args):
    def possible_end(s):
        word = "[Instruction]"
        for i in range(len(word) - 1, 0, -1):
            if s[-i:] == word[:i]:
                return 0
        if s[-len(word) :] == word:
            return 1
        return -1

    def generate(question):
        x = mx.array([[tokenizer.bos_id()] + tokenizer.encode(question)])
        skip = 0
        prompt_processing = None
        tokens = []
        start = tic()
        for token in model.generate(x, args.temp):
            tokens.append(token)

            if len(tokens) == 1:
                # Actually perform the computation to measure the prompt processing time
                mx.eval(token)
                prompt_processing = toc("Prompt processing", start)

            if len(tokens) >= args.num_tokens:
                break

            mx.eval(tokens)
            token_list = [t.item() for t in tokens]
            s = tokenizer.decode(token_list)

            end = possible_end(s)
            if end == 0:
                continue
            if end == 1:
                skip = len(s)
                break

            print(s[skip:], end="", flush=True)
            skip = len(s)
            if token_list[-1] == tokenizer.eos_id():
                break

        mx.eval(tokens)
        full_gen = toc("Full generation", start)
        s = tokenizer.decode([t.item() for t in tokens])
        print(s[skip:], end="", flush=True)

    prompt = open(args.prompt).read().strip()
    while True:
        question = input("Ask a question: ")
        generate(prompt.replace("{}", question))
        print()


def load_model(model_path):
    weights = mx.load(model_path)
    mlp_dims, dims = weights["layers.0.linear1.weight"].shape
    num_heads = dims // 128
    num_layers = max(int(l.split(".")[1]) for l in weights.keys() if "layers" in l) + 1
    vocab_size = weights["out_proj.weight"].shape[-1]
    model = Llama(num_layers, vocab_size, dims, mlp_dims, num_heads)
    model.update(tree_unflatten(list(weights.items())))
    mx.eval(model.parameters())
    return model

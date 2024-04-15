import time
from recurrentgemma import jax as recurrentgemma
import pathlib
import sentencepiece as spm
import jax.numpy as jnp
import logging
import os

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true "
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_async_collectives=true "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
    "--xla_gpu_enable_highest_priority_async_stream=true "
)

MODEL = "2b-it"
WEIGHTS_DIR = f"/home/dom/.cache/kagglehub/models/google/recurrentgemma/flax/{MODEL}/1"

weights_dir = pathlib.Path(WEIGHTS_DIR)
ckpt_path = weights_dir / MODEL
vocab_path = weights_dir / "tokenizer.model"

params = recurrentgemma.load_parameters(ckpt_path, "single_device")

vocab = spm.SentencePieceProcessor()
vocab.Load(str(vocab_path))

config = recurrentgemma.GriffinConfig.from_flax_params_or_variables(params)
model = recurrentgemma.Griffin(config, dtype=jnp.bfloat16)

sampler = recurrentgemma.Sampler(model=model, vocab=vocab, params=params)


def toks_per_sec(sampler, prompts, total_generation_steps):
    # jit step
    sampler(input_strings=prompts, total_generation_steps=total_generation_steps)

    # real timing
    t0 = time.time()
    out = sampler(input_strings=prompts, total_generation_steps=total_generation_steps)
    t1 = time.time()
    # divide the number of tokens by the time taken
    n_tokens = sum(len(toks) for toks in out.tokens)
    timing = n_tokens / (t1 - t0)
    print(f"{timing:.1f} tokens/sec")
    return timing, out


prompts = [
    "The president of the United States is",
    "The recipe for a delicious cake is a",
    "write a long fiction novel. do not be lazy. delve into all the details. go further into small things far more than you want to. be pandantic.",
    "The best cartoon in the world is",
    "What is the meaning of life?",
    "Biology is the study of",
    "The best way to cook a steak is",
    "Python is a programming language that is",
]

max_sequence_length = 1024
timing, out = toks_per_sec(sampler, prompts, max_sequence_length)

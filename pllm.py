import time
import torch
from recurrentgemma import torch as recurrentgemma
import pathlib
import sentencepiece as spm
import logging

MODEL = "2b-it"
WEIGHTS_DIR = (
    f"/home/dom/.cache/kagglehub/models/google/recurrentgemma/pyTorch/{MODEL}/1"
)

weights_dir = pathlib.Path(WEIGHTS_DIR)
ckpt_path = weights_dir / f"{MODEL}.pt"
vocab_path = weights_dir / "tokenizer.model"

device = "cuda" if torch.cuda.is_available() else "cpu"

params = torch.load(ckpt_path)
params = {k: v.to(device=device) for k, v in params.items()}

vocab = spm.SentencePieceProcessor()
vocab.Load(str(vocab_path))

config = recurrentgemma.GriffinConfig.from_torch_params(params)
model = recurrentgemma.Griffin(config, dtype=torch.bfloat16, device=device)
model.load_state_dict(params)
model = torch.compile(model)

sampler = recurrentgemma.Sampler(model=model, vocab=vocab)


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
]
timing, out = toks_per_sec(sampler, prompts, 256)

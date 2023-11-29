import torch
from torch.nn import functional as F


def compute_perplexity(prompt, result):
    return 0


def evaluate_model(model, prompt):
    model.eval()
    with torch.no_grad():
        tokenized_prompt = prompt
        tokenized_result = model.generate(tokenized_prompt)
        result = tokenized_result
        perplexity = compute_perplexity(prompt, result)

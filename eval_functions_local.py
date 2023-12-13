from sklearn.model_selection import KFold
import torch
from torch.nn import functional as F

def tokenize_text(text):
    """
    Tokenize un texte au niveau des caractères.

    Args:
    - text (str): Le texte à tokenizer.

    Returns:
    - List[int]: Une liste d'entiers représentant les indices des caractères tokenisés.
    """
    tokens = [ord(char) for char in text]
    return tokens


def split_text(text, train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1):
    """
    Divise un texte brut en trois parties : entraînement, devset, test.

    Args:
    - text (str): Le texte brut à diviser.
    - train_ratio (float): Proportion du texte à utiliser pour l'entraînement.
    - dev_ratio (float): Proportion du texte à utiliser pour le devset.
    - test_ratio (float): Proportion du texte à utiliser pour le test.

    Returns:
    - tuple: Un tuple contenant trois strings représentant les parties d'entraînement, devset, et test.
    """
    total_length = len(text)
    train_end = int(total_length * train_ratio)
    dev_end = int(total_length * (train_ratio + dev_ratio))

    train_text = text[:train_end]
    dev_text = text[train_end:dev_end]
    test_text = text[dev_end:]

    return train_text, dev_text, test_text

def compute_perplexity(prompt, result):
    return 0


def evaluate_model(model, prompt):
    model.eval()
    with torch.no_grad():
        tokenized_prompt = prompt
        tokenized_result = model.generate(tokenized_prompt)
        result = tokenized_result
        perplexity = compute_perplexity(prompt, result)


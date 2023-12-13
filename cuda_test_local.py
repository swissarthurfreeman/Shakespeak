import torch

def check_cuda_availability():
    """
    Vérifie si CUDA est disponible et imprime un message correspondant.
    """
    if torch.cuda.is_available():
        print("CUDA est disponible. Vous pouvez utiliser le GPU.")
    else:
        print("CUDA n'est pas disponible. Utilisation du CPU.")

# Appel de la fonction pour vérifier la disponibilité de CUDA
check_cuda_availability()

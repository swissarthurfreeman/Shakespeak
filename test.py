import math

# Liste de nombres à multiplier
nombres = [2, 3, 5, 7, 11]

# Calcul de la multiplication en utilisant les logarithmes
log_produit = sum(math.log(x) for x in nombres)

# Calcul de l'exponentielle pour obtenir le produit
produit = math.exp(log_produit)

print("Produit sans utiliser les logarithmes :", math.prod(nombres))
print("Produit en utilisant les logarithmes :", produit)
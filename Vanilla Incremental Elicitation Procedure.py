import numpy as np
from mip import Model, xsum, BINARY, minimize, maximize

# Fonction pour calculer le modèle de décision avec les poids
def M_omega(x, omega):
    return np.dot(x, omega)

# Fonction pour poser une question au décideur
def Query(x, y):
    # Simule la préférence du décideur (choix aléatoire ici, à remplacer par une vraie préférence)
    print(f"Comparaison entre {x} et {y}")
    return x if np.random.rand() > 0.5 else y

# Fonction pour calculer le regret maximum par paires en tenant compte des contraintes de Omega
def PMR(x, y, Omega):
    # Créer un modèle mip pour résoudre le problème linéaire
    m = Model(sense=maximize)
    
    m_crit = len(x)  # Nombre de critères
    
    # Variables de décision : poids w_i pour chaque critère
    w = [m.add_var(lb=0) for _ in range(m_crit)]
    
    # Contrainte : somme des poids w_i = 1 (normalisation des poids)
    m += xsum(w[i] for i in range(m_crit)) == 1
    
    # Contrainte : ajouter les contraintes définies par le polytope Omega
    for constraint in Omega:
        m += xsum(w[i] * constraint['coefficients'][i] for i in range(m_crit)) >= constraint['rhs']
    
    # Objectif : maximiser M_omega(y) - M_omega(x)
    m.objective = xsum(w[i] * (y[i] - x[i]) for i in range(m_crit))
    
    # Résoudre le problème
    m.optimize()
    
    # Obtenir le regret maximal
    return m.objective_value

# Fonction pour mettre à jour l'ensemble des poids possibles (Omega) en fonction de la réponse
def Update(Omega, answer, x, y):
    # On ajoute une contrainte linéaire en fonction de la préférence entre x et y
    if np.array_equal(answer, x):
        # x est préféré à y -> on ajoute la contrainte M_omega(x) >= M_omega(y)
        new_constraint = lambda omega: np.dot(omega, x) >= np.dot(omega, y)
    else:
        # y est préféré à x -> on ajoute la contrainte M_omega(y) >= M_omega(x)
        new_constraint = lambda omega: np.dot(omega, y) >= np.dot(omega, x)
    
    # On applique cette contrainte à l'ensemble des poids Omega
    Omega = [omega for omega in Omega if new_constraint(omega)]
    
    return Omega

# Fonction pour calculer le regret maximum (MR) pour une solution x par rapport à toutes les autres solutions
def MR(x, X, Omega):
    return max(PMR(x, y, Omega) for y in X if not np.array_equal(x, y))

# Fonction pour calculer le regret minimax (mMR) et obtenir les solutions x_star et y_star
def mMR(X, Omega):
    x_star = min(X, key=lambda x: MR(x, X, Omega))  # Solution minimisant le regret max
    y_star = max(X, key=lambda y: PMR(x_star, y, Omega))  # Solution maximisant le regret par rapport à x_star
    max_regret = MR(x_star, X, Omega)
    return max_regret, x_star, y_star

# Algorithme Vanilla Incremental Elicitation Procedure
def vanilla_incremental_elicitation(X, Omega, epsilon):
    # Calculer le regret minimax initial
    max_regret, x_star, y_star = mMR(X, Omega)
    
    # Boucle jusqu'à ce que le regret minimax soit inférieur à epsilon
    while max_regret >= epsilon:
        print(f"Regret actuel: {max_regret}")
        print(f"Question: préférez-vous {x_star} ou {y_star} ?")
        
        # Poser la question au décideur et obtenir la réponse
        answer = Query(x_star, y_star)
        
        # Mettre à jour l'ensemble des poids Omega en fonction de la réponse
        Omega = Update(Omega, answer, x_star, y_star)
        
        # Recalculer le regret minimax et les solutions x_star et y_star
        max_regret, x_star, y_star = mMR(X, Omega)
    
    # Retourner la solution finale
    return x_star

# Exemple de données
def exemple_elicitation():
    # Ensemble des solutions (3 solutions avec 2 critères chacune)
    X = np.array([[0.5, 0.2], [0.7, 0.1], [0.6, 0.3]])

    # Ensemble initial de poids possibles (Omega) [on commence avec 5 vecteurs de poids aléatoires normalisés]
    # Polytope initial : normalisation des poids
    Omega = [{'coefficients': [1, 1], 'rhs': 1}]  # Initialisation d'un simple polytope

    # Paramètre de tolérance pour le regret minimax
    epsilon = 0.1

    print("Initialisation de l'algorithme d'élucidation incrémentale avec programme linéaire.\n")

    # Lancer l'algorithme
    solution = vanilla_incremental_elicitation(X, Omega, epsilon)

    print(f"\nSolution finale recommandée: {solution}")

# Lancer l'exemple
exemple_elicitation()


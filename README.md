Yolah AI – Minimax + reseau de neurone +AlphaZero en methode d'apprentissage , je genere des datasets que j'entraine dessus


en gros je renforce avec un minmax depth=1 vs un minmax depth=1 .
Au début l'IA joue avec un minimax classique + une heuristique, puis j’ai ajouté un réseau de neurones pour essayer d’améliorer les décisions.

L’idée globale c’est :

on fait jouer l’IA toute seule (avec un reseau de neurone nulle au début donc on utilise seulement l'heuristique)
on récupère des données (positions + résultat)
on entraine un modele x
et après on remplace l’évaluation par le modèle 

C’est pas parfait mais ça donne déjà des résultats corrects.

Installer le projet

Cloner :

git clone <repo_url>
cd Yolah

Créer un environnement (c important) :

python3 -m venv tf-env
source tf-env/bin/activate

Installer les libs :

pip install tensorflow-cpu numpy scikit-learn

Lancer une partie

python main.py

Dans main.py tu peux changer la profondeur :

depth=1 cest plus rapide
depth=2 cest meilleur ca renforce l'ia mais plus lent

Entrainer l’IA
Générer les données

python -m ai.generate_data

Ça va créer :

data/yolah_dataset.csv

En gros l’IA joue contre elle-même avec un mix random + minimax.

Entrainer le modèle

python ai/train_nn.py

Ça crée :

data/yolah_value_model.keras

C’est le modèle utilisé ensuite.

Comment j’ai bossé dessus

J’ai fait ça en plusieurs étapes :

minimax + heuristique
génération de parties
entrainement du NN
remplacer l’évaluation par le NN
refaire un peu de génération + train

Mais faut pas trop boucler sinon le modèle apprend ses propres erreurs.(genre on train aux max 2 fois sinon le score il va donner -1 et genre ça perd beaucoup après)

Paramètres utiles

Dans generate_data.py :

num_games = 5000
depth = 1
random_ratio = 0.3

Plus de parties = mieux mais plus long
Depth élevé = meilleur mais ça ralentit beaucoup

si vous avez des questions dites moi

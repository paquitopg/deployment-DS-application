"""
This is a module docstring
"""

import os
import argparse
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix
import duckdb

os.chdir("/home/onyxia/work/deployment-DS-application")
titanic = pd.read_csv("data.csv")


con = duckdb.connect(database=":memory:")

# Check la structure de Name "Nom, Prénom"
bad = con.sql("""
    SELECT COUNT(*) AS n_bad
    FROM titanic
    WHERE list_count(string_split(Name, ',')) <> 2
""").fetchone()[0]

if bad == 0:
    print("Test 'Name' OK se découpe toujours en 2 parties avec ','")
else:
    print(f"Problème dans la colonne Name: {bad} ne se décomposent pas en 2 parties.")


parser = argparse.ArgumentParser(description="How many trees?")
parser.add_argument(
    "--n_trees", type=int, default=20, help="A number of trees to choose"
)
args = parser.parse_args()

N_TREES = args.n_trees
print(N_TREES)
MAX_DEPTH = None
MAX_FEATURES = "sqrt"


## Encoder les données imputées ou transformées.

numeric_features = ["Age", "Fare"]
categorical_features = ["Embarked", "Sex"]

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", MinMaxScaler()),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder()),
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        ("Preprocessing numerical", numeric_transformer, numeric_features),
        (
            "Preprocessing categorical",
            categorical_transformer,
            categorical_features,
        ),
    ]
)

pipe = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=N_TREES)),
    ]
)


# splitting samples
y = titanic["Survived"]
X = titanic.drop("Survived", axis="columns")

# On _split_ notre _dataset_ d'apprentisage pour faire de la
# validation croisée une partie pour apprendre une partie pour regarder le score.
# Prenons arbitrairement 10% du dataset en test et 90% pour l'apprentissage.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# check que pas de problème de data leakage
if set(X_train["Embarked"].dropna().unique()) - set(
    X_test["Embarked"].dropna().unique()
):
    message = "Problème de data leakage pour la variable Embarked"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)

if set(X_train["Sex"].dropna().unique()) - set(X_test["Sex"].dropna().unique()):
    message = "Problème de data leakage pour la variable Sex"
else:
    message = "Pas de problème de data leakage pour la variable Embarked"

print(message)


JETONAPI = "$trotskitueleski1917"

# Vérifie les valeurs manquantes
# TODO: généraliser à toutes les variables
n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Survived IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Survived"
message_warn = f"{n_missing} valeurs manquantes pour la variable Survived"
message = message_ok if n_missing == 0 else message_warn
print(message)

n_missing = con.sql("""
    SELECT COUNT(*) AS n_missing
    FROM titanic
    WHERE Age IS NULL
""").fetchone()[0]

message_ok = "Pas de valeur manquante pour la variable Age"
message_warn = f"{n_missing} valeurs manquantes pour la variable Age"
message = message_ok if n_missing == 0 else message_warn
print(message)


# Random Forest
# Ici demandons d'avoir 20 arbres
pipe.fit(X_train, y_train)


# calculons le score sur le dataset d'apprentissage et sur le
# dataset de test (10% du dataset d'apprentissage mis de côté)
# le score étant le nombre de bonne prédiction
rdmf_score = pipe.score(X_test, y_test)
rdmf_score_tr = pipe.score(X_train, y_train)
print(f"{rdmf_score:.1%} de bonnes réponses sur les données de test pour validation")

print(20 * "-")
print("matrice de confusion")
print(confusion_matrix(y_test, pipe.predict(X_test)))

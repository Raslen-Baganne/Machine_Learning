import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Création du DataFrame initial
data = {
    "Groupe": [1, 2, 3],
    "nombreEtudiant": [20, 10, 18]
}

df = pd.DataFrame(data)

# Affichage du DataFrame
print("DataFrame initial :")
print(df)

# Affichage des types des colonnes
print("\nType des colonnes :")
print(df.dtypes)

# Accès à la première ligne du DataFrame
print("\nPremière ligne du DataFrame :")
print(df.loc[0])

# Accès aux lignes 0 et 2
print("\nAccès aux lignes 0 et 2 :")
print(df.loc[[0, 2]])

# Calcul de la médiane
print("\nMédiane :")
print(df.median())

# Calcul de la moyenne
print("\nMoyenne :")
print(df.mean())

# Valeurs maximales
print("\nValeurs maximales :")
print(df.max())

# Somme des colonnes
print("\nSomme des colonnes :")
print(df.sum())

# Exportation du DataFrame en CSV
df.to_csv("test.csv", sep=';', index=False, header=False)

# Lecture d'un autre CSV
new_df = pd.read_csv("exemple.csv", sep=';')

# Affichage du nouveau DataFrame chargé depuis le CSV
print("\nNouveau DataFrame chargé depuis exemple.csv :")
print(new_df)

# Chargement du DataFrame Titanic
df = pd.read_csv("titanic.csv")
print("\nDataFrame Titanic :")
print(df)

# a) Afficher les 10 premières lignes du DataFrame df
print("\n10 premières lignes du DataFrame Titanic :")
print(df.head(10))

# b) Afficher les 10 dernières lignes du DataFrame df
print("\n10 dernières lignes du DataFrame Titanic :")
print(df.tail(10))

# Informations sur le DataFrame
print("\nShape du DataFrame :")
print(df.shape)
print("\nInfo du DataFrame :")
print(df.info())
print("\nStatistiques descriptives du DataFrame :")
print(df.describe())
print("\nColonnes du DataFrame :")
print(df.columns)

# Suppression des colonnes inutiles
df = df.drop(columns=["SibSp", "Parch", "Ticket", "Cabin", "Embarked"])
df.set_index("PassengerId", inplace=True)

# Affichage des valeurs manquantes
print("\nValeurs manquantes par colonne :")
print(df.isnull().sum())

# Remplissage des valeurs manquantes dans la colonne Age
df["Age"].fillna(df.groupby("Pclass")["Age"].transform("median"), inplace=True)

# Extraction du titre des passagers
df["Title"] = df['Name'].str.extract('([A-Za-z]+)\.', expand=False)
title_counts = df["Title"].value_counts()
print("\nNombre de titres :")
print(title_counts)

# Mapping des titres
title_mapping = {
    "Mr": 0, "Miss": 1, "Mrs": 2,
    "Master": 3, "Dr": 3, "Rev": 3,
    "Col": 3, "Major": 3, "Mlle": 3,
    "Mme": 3, "Capt": 3, "Sir": 3,
    "Lady": 3, "Jonkheer": 3, "Don": 3,
    "Dona": 3, "Countess": 3, "Ms": 3
}

df['Title'] = df["Title"].map(title_mapping)

# Changer les valeurs de la colonne Sex
sex_value={"male":0,"female":1}
df['Sex']=df['Sex'].map(sex_value)

# Afficher le DataFrame après le changement de la colonne Sex
print("\nDataFrame après le changement de la colonne Sex :")
print(df[['Sex', 'Title']])

# Afficher les passagers ayant une valeur de Fare égale à 7.25
passagers_fare_7_25 = df.loc[df['Fare'] == 7.25]

# Afficher les passagers avec un Fare de 7.25
print("\nPassagers ayant une valeur de Fare égale à 7.25 :")
print(passagers_fare_7_25)

# Modification des valeurs de la colonne Fare selon les critères spécifiés
df.loc[df['Fare']<=17,'Fare']=0
df.loc[(df['Fare']>17) & (df['Fare']<=30),'Fare']=1
df.loc[(df['Fare']>30) & (df['Fare']<=100),'Fare']=2
df.loc[df['Fare']>100,'Fare']=3

# Afficher le DataFrame après le changement de la colonne Fare
print("\nDataFrame après le changement de la colonne Fare :")
print(df['Fare'])

X_simple = df[['Fare']]
y_simple = df['Survived']

# Diviser les données en ensembles d'entraînement et de test
X_train_simple, X_test_simple, y_train_simple, y_test_simple = train_test_split(X_simple, y_simple, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train_simple)

# Faire des prédictions
y_pred_simple = model_simple.predict(X_test_simple)

# Évaluer le modèle
mse_simple = mean_squared_error(y_test_simple, y_pred_simple)
r2_simple = r2_score(y_test_simple, y_pred_simple)

print("Évaluation du modèle de régression linéaire simple :")
print(f"MSE : {mse_simple}")
print(f"R² : {r2_simple}")
print("Coefficient de la variable Fare :", model_simple.coef_)
print("Ordonnée à l'origine :", model_simple.intercept_)

# Étape 2 : Régression Linéaire Multiple
# Sélectionner les colonnes pour la régression linéaire multiple
X_multi = df[['Fare', 'Sex', 'Age', 'Title']]
y_multi = df['Survived']

# Diviser les données en ensembles d'entraînement et de test
X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(X_multi, y_multi, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model_multi = LinearRegression()
model_multi.fit(X_train_multi, y_train_multi)

# Faire des prédictions
y_pred_multi = model_multi.predict(X_test_multi)

# Évaluer le modèle
mse_multi = mean_squared_error(y_test_multi, y_pred_multi)
r2_multi = r2_score(y_test_multi, y_pred_multi)

print("\nÉvaluation du modèle de régression linéaire multiple :")
print(f"MSE : {mse_multi}")
print(f"R² : {r2_multi}")
print("Coefficients des variables :")
for i, col in enumerate(X_multi.columns):
    print(f"{col}: {model_multi.coef_[i]}")
print("Ordonnée à l'origine :", model_multi.intercept_)

print(f"Dimensions de X_test_multi : {X_test_multi.shape}")
print(f"Dimensions de y_test_multi : {y_test_multi.shape}")

# Visualisation des résultats de la régression linéaire simple
plt.figure(figsize=(10, 6))
plt.scatter(X_test_simple, y_test_simple, color='blue', label='Données Réelles')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Prédictions')
plt.title("Régression Linéaire Simple : Prédictions de Survie")
plt.xlabel("Fare")
plt.ylabel("Survived")
plt.legend()
plt.grid()
plt.show()

# Visualisation des résultats de la régression linéaire multiple
plt.figure(figsize=(10, 6))
plt.scatter(X_test_multi, y_test_multi, color='blue', label='Valeurs Réelles')
plt.scatter(X_test_multi, y_pred_multi, color='red', label='Prédictions', alpha=0.5)
plt.plot(X_test_multi, y_pred_multi, color='green', linewidth=2, label='Ligne de Régression')
plt.title("Visualisation des Résultats de la Régression Linéaire Multiple")
plt.xlabel("Caractéristiques (X)")
plt.ylabel("Cible (y)")
plt.legend()
plt.grid()
plt.show()
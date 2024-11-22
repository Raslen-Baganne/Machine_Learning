import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import preprocessing
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from math import sqrt

# Chargement et exploration des données
data = pd.read_csv('ozone.csv')
print(data.head())
print(data.info())

# Suppression des colonnes inutiles
data = data.drop(columns=['Vent', 'Pluie', 'Date'])
print(data.head())

# Vérification des valeurs manquantes
missing_values = data.isnull().sum()
print("Valeurs manquantes par colonne :\n", missing_values)

# Fonction de normalisation
def normalization(data):
    for col in data.columns:
        x = data[[col]].values.astype(float)
        standard_normalization = preprocessing.StandardScaler()
        res = standard_normalization.fit_transform(x)
        data[[col]] = res
        print(data[[col]])

# Séparation des données explicatives et de la variable cible
X = data[['T9', 'T12', 'T15', 'Ne9', 'Ne12', 'Ne15', 'Vx9', 'Vx12', 'Vx15', 'MaxO3v']]
y = data['MaxO3']

# Normalisation des données explicatives
normalization(X)

# Boucle d'essai avec la régression linéaire et calcul du RMSE et R²
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_test_pred = model.predict(X_test)
    RMSE = sqrt(mean_squared_error(y_test, y_test_pred))  # Racine carrée du MSE
    print(f"Essai {i + 1}")
    print("Ensemble de test - RMSE:", RMSE)
    print("Ensemble de test - MSE:", mean_squared_error(y_test, y_test_pred))
    print("Ensemble de test - Score R²:", r2_score(y_test, y_test_pred))
    print("-------------")

# Utilisation de KFold pour la validation croisée
kf = KFold(n_splits=4, shuffle=True, random_state=42)
for train_index, test_index in kf.split(X):
    print("Les indices de train index", train_index)
    print("Les indices de test index", test_index)
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Modélisation avec la régression linéaire
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Prédiction et évaluation
    y_test_pred = model.predict(X_test)
    RMSE = sqrt(mean_squared_error(y_test, y_test_pred))
    print("Résultats de la validation croisée K-Fold :")
    print("Test des performances du modèle sur les ensembles de test")
    print("-" * 50)
    print(f"  - RMSE de l'ensemble de test : {RMSE:.4f}")
    print(f"  - MSE de l'ensemble de test  : {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"  - Score R² de l'ensemble de test : {r2_score(y_test, y_test_pred):.4f}")
    print("-" * 50)
    print("--------------------------------")


    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_r2_scores = []
test_r2_scores = []

for i in range(1, 11):
    poly = PolynomialFeatures(degree=i)#par exemple:features = x1 et x2 yemchi yzid (x1)**2 et (x2)**2 et (x1)*(x2) rq:bl i ywali kol marra yakhou degré (1,2,3,4...)
    poly_x_train = poly.fit_transform(X_train)
    poly_x_test = poly.transform(X_test)
    
   
    model = LinearRegression()
    model.fit(poly_x_train, y_train)
    
    
    y_train_pred = model.predict(poly_x_train)
    y_test_pred = model.predict(poly_x_test)
    
    
    RMSE_train = sqrt(mean_squared_error(y_train, y_train_pred))
    MSE_train = mean_squared_error(y_train, y_train_pred)
    R2_train = r2_score(y_train, y_train_pred)
    
    
    RMSE_test = sqrt(mean_squared_error(y_test, y_test_pred))
    MSE_test = mean_squared_error(y_test, y_test_pred)
    R2_test = r2_score(y_test, y_test_pred)
    
    
    print(f"Degré polynomial: {i}")
    print(f"  - Entraînement | RMSE: {RMSE_train:.4f}, MSE: {MSE_train:.4f}, R²: {R2_train:.4f}")
    print(f"  - Test        | RMSE: {RMSE_test:.4f}, MSE: {MSE_test:.4f}, R²: {R2_test:.4f}")
    print("-" * 50)
    
    train_r2_scores.append(R2_train)
    test_r2_scores.append(R2_test)
    
plt.figure(figsize=(10, 6))
plt.plot(range(1, 4), train_r2_scores[:3], label="Score R² - Entraînement", marker='o', color='blue')
plt.plot(range(1, 4), test_r2_scores[:3], label="Score R² - Test", marker='o', color='red')
plt.xlabel("Degré polynomial")
plt.ylabel("Score R²")
plt.title("Score R² en fonction du degré polynomial")
plt.legend()
plt.grid(True)
plt.show()


    
    
    
    
    








    

    

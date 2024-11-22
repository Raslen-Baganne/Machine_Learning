import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score 
import matplotlib.pyplot as plt
import seaborn as sns 

# Lire le fichier Excel
data = pd.read_excel('maisons.xlsx')  # Remplacez 'nom_du_fichier.xlsx' par le nom réel de votre fichier

# Afficher les données

print(data)

#Supprimer la colonne date 
data.drop(columns=["Date", "No"],inplace=True) 
print(data)

# une fonction pour binariser une colonne  
# donnee:le dataframe  
# cl: le nom de la colonne  
def binariser(donnee,cl): 
    #Sélectionner la colonne et calculer la moyenne  
    moy= donnee[cl].mean()  
    print(moy) 
    # Remplacer les valeurs supérieures à la moyenne par 1 et le reste par 0  
    donnee[cl] =(donnee[cl] >moy).astype(float)  
# binariser latitude  
binariser(data,"Latitude")  
# binariser longitude  
binariser(data,"Longitude") 

# Séparer les données en entrées et sorties 
X =data.iloc[:,:-1] #les caractéristiques (Features) 
y =data.iloc[:,-1] #les résultats (classes, Target) 

X_train, X_test, y_train,y_test=train_test_split(X,y,test_size=0.2) 

model= LinearRegression() 
# Entrainer un modèle de régression linéaire simple(age) 
model.fit(X_train[["Age"]],y_train)  
# Afficher les coefficients  
print("Coefficients regression simple: ", model.coef_, " w0= ", model.intercept_) #a hiye model.coef w b hiye model.intercept w hiye les valeurs mte3 el test mte3 el age(ax+b)
# Prédire les résultats des échantillons de test 
y_pred=model.predict(X_test[["Age"]])

# Evaluation du modèle 
print("Régression simple: MSE = ", mean_squared_error(y_test, y_pred)) 
print("Score R2=",r2_score(y_test,y_pred)) 

plt.scatter(X_test["Age"],y_test, color="black") 
plt.plot(X_test["Age"],y_pred, color='r') 
plt.legend(["linéaire"]) 
plt.xlabel("Age") 
plt.ylabel("prix") 
plt.grid() 
plt.show() 

modelRegMulti=LinearRegression() 
modelRegMulti.fit(X_train, y_train) 
print("Coefficients régression multiple: ",modelRegMulti.coef_," w0= ", 
modelRegMulti.intercept_) 
yl_predm =modelRegMulti.predict(X_test) 
print("Régression multiple:MSE= ",mean_squared_error(y_test,yl_predm)) 
print("Score R2m:",r2_score(y_test, yl_predm))

df_corr = X.corr() 
ax = sns.heatmap(df_corr,cmap = 'coolwarm') 



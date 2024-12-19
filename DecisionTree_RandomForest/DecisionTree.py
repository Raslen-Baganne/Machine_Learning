from sklearn import tree
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import plot_tree
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier 


path = "../TP ML/heart.csv"
df = pd.read_csv(path)
print(df)
print(df.hist())

plt.figure(figsize=(8, 5))
sns.histplot(df['age'], kde=True, bins=20, color='blue')
plt.title('Distribution de l\'âge')
plt.show()

sns.countplot(df['age'], hue='',data = df)

plt.figure(figsize=(8, 5))
sns.scatterplot(x='thalach', y='target', data=df, hue='sex', palette='coolwarm')
plt.title('Fréquence cardiaque maximale vs cible')
plt.show()


X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)



abr = DecisionTreeClassifier(max_depth=3, min_samples_leaf=2)
abr.fit(X_train, y_train)


y_pred = abr.predict(X_test)


fig = plt.figure(figsize=(15,7)) 
fn = ["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]  
cn = ['0', '1']  
tree.plot_tree(abr, feature_names =fn, class_names=cn, filled=True)  
plt.show() 


accuracy = accuracy_score(y_test, y_pred)  
error_rate = 1 - accuracy  

print("Accuracy Score:", accuracy)
print("Error Rate:", error_rate)

importance = pd.DataFrame({'feature': X_train.columns,'importance' : np.round(abr.feature_importances_,3)}) 
importance.sort_values('importance', ascending=False, inplace = True) 
print(importance)
best_score = 0
best_params = {'max_depth': None, 'min_samples_leaf': None}
max_depth_values=[1,2,4,6,8,10,12]
min_samples_leaf_values=[2,3,5,10,15,20]
for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
        model.fit(X_train, y_train)
       
        y_pred = model.predict(X_test)
       
        score = accuracy_score(y_test, y_pred)
       
        if (score>best_score):
            best_score=score
            best_params['max_depth']=max_depth#el max howa ekher star fel arbre(4 start houwa el max depth)
            best_params['min_samples_leaf']=min_samples_leaf#


print("best score is :",best_score)
print("best max_depth value is ",best_params['max_depth'])
print("best min_samples_leaf value is ",best_params['min_samples_leaf'])



optimized_tree = DecisionTreeClassifier(max_depth=best_params['max_depth'], min_samples_leaf=best_params['min_samples_leaf'])
optimized_tree.fit(X_train, y_train)

plt.figure(figsize=(15, 7))
plot_tree(optimized_tree, 
          feature_names=X.columns, 
          class_names=['0', '1'], 
          filled=True)
plt.title("Arbre de décision optimisé")
plt.show()


optimized_y_pred = optimized_tree.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_y_pred)
print("Accuracy du modèle optimisé :", optimized_accuracy)

max_depth_values = [1, 2, 4, 6, 8, 10, 12]
min_samples_leaf_values = [2, 3, 5, 10, 15, 20]


best_score = 0
best_params = {'max_depth': None, 'min_samples_leaf': None}

for max_depth in max_depth_values:
    for min_samples_leaf in min_samples_leaf_values:
        
        model = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=42)
        
        
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mean_score = scores.mean() 
       
        if mean_score > best_score:
            best_score = mean_score
            best_params['max_depth'] = max_depth
            best_params['min_samples_leaf'] = min_samples_leaf


print("Meilleurs paramètres trouvés avec validation croisée :")
print("max_depth :", best_params['max_depth'])
print("min_samples_leaf :", best_params['min_samples_leaf'])
print("Meilleur score de validation croisée :", best_score)


optimized_tree = DecisionTreeClassifier(
    max_depth=best_params['max_depth'],
    min_samples_leaf=best_params['min_samples_leaf'],
    random_state=42
)
optimized_tree.fit(X_train, y_train)

plt.figure(figsize=(15, 7))
plot_tree(optimized_tree, 
          feature_names=X.columns, 
          class_names=['0', '1'], 
          filled=True)
plt.title("Arbre de décision optimisé")
plt.show()

# Calculer et afficher l'accuracy du modèle final sur les données de test
optimized_y_pred = optimized_tree.predict(X_test)
optimized_accuracy = accuracy_score(y_test, optimized_y_pred)
print("Accuracy du modèle optimisé sur les données de test :", optimized_accuracy)




X = df.drop('target', axis=1)  
y = df['target'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

FRcl = RandomForestClassifier(n_estimators=100, random_state=42)

FRcl.fit(X_train, y_train)

y_pred = FRcl.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy

print("Accuracy Score:", accuracy)
print("Error Rate:", error_rate)
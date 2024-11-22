import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

path = "http://mob.u-strasbg.fr/lab/data/titanic.csv"
df = pd.read_csv(path)

print("Aperçu des données :")
print(df.head())

plt.figure(figsize=(8, 6))
sns.countplot(x="Survived", data=df)
plt.title("Distribution des survivants")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x="Survived", hue="Sex", data=df)
plt.title("Survivants par sexe")
plt.show()

plt.figure(figsize=(8, 6))
sns.countplot(x="Survived", hue="Pclass", data=df)
plt.title("Survivants par classe")
plt.show()


plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
plt.title("Carte des valeurs manquantes")
plt.show()


colonnes_a_supprimer = ['Ticket', 'Cabin', 'Name', 'Embarked']
df = df.drop(columns=colonnes_a_supprimer)


print("Colonnes restantes après suppression :")
print(df.columns)

df.set_index('PassengerId', inplace=True)
df['Age'] = df['Age'].fillna(df.groupby("Pclass")['Age'].transform("median"))#ya5ou groupby el Pclass w tchouf eli m3ndhomch age t3amarhom bl valeur el westaniya exmple(ely 3andhom Pclass = 1 lkol nchoufou el valeur el wostaniye mte3 el age w n7otoha feli m3ndouch age w tebde nefs el Pclass(=1))

print("Valeurs manquantes dans la colonne Age après traitement :", df['Age'].isnull().sum())
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
print("Valeurs uniques dans la colonne Sex :", df['Sex'].unique())
df.dropna(inplace=True)
X = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy Score :", accuracy_score(y_test, y_pred))
print("\nClassification Report:", classification_report(y_test, y_pred))
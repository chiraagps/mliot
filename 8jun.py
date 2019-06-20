import pandas as pd 
dataset = pd.read_csv('iris.csv')

x = dataset.iloc[:,1:5].values
y = dataset.iloc[:,5].values

from sklearn.preprocessing import LabelEncoder
lEncoder = LabelEncoder()
y = lEncoder.fit_transform(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)
from sklearn.tree import DecisionTreeClassifier


dClassifier = DecisionTreeClassifier()
dClassifier.fit(x_train,y_train)


y_pred = dClassifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

import matplotlib.pyplot as plt
import seaborn as sn

plt.figure(figsize=(3,3))
sn.heatmap(cm,annot=True)


from sklearn.tree import export_graphviz
export_graphviz(dClassifier,out_file='iris.dot',
                feature_names = ['1','2','3','4'],
                class_names=['1','2','3'],filled=True)

from subprocess import call

call(['dot','-Tpng','iris.dot','-o','Iris.png','-Gdpi=600'])

from IPython.display import Image

Image(filename = 'Iris.png')












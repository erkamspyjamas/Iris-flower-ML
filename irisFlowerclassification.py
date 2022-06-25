

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import  DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

iris = pd.read_csv("Iris.csv") #Iris.csv dosyasini pandas dataframe'ine aktardim.

x = iris.iloc[:, :-1].values
y = iris.iloc[:, -1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)  #Veri setini egitim ve test kumelerine boldurdum
classifier   = DecisionTreeClassifier()
classifier.fit(x_train,   y_train) #burada egitiyorum
y_pred   = classifier.predict(x_test) #tahminler urettiriyorum

iris_data=load_iris()   #iris veri setini kutuphaneden cekiyorum
iris_df = pd.DataFrame(iris_data.data, columns = iris_data.feature_names) #dataframe olusturuyorum

kmeans = KMeans(n_clusters=3,init = 'k-means++',   max_iter = 100, n_init = 10, random_state = 0) #KMeans algoritmasini uyguluyorum
y_kmeans = kmeans.fit_predict(x)

plt.scatter(x[y_kmeans   == 0, 0], x[y_kmeans == 0, 1],s = 100, c = 'red', label = 'Iris-setosa')
plt.scatter(x[y_kmeans   == 1, 0], x[y_kmeans == 1, 1],s = 100, c = 'blue', label = 'Iris-versicolour')
plt.scatter(x[y_kmeans   == 2, 0], x[y_kmeans == 2, 1],s = 100, c = 'green', label = 'Iris-virginica')   #Kumeleri gorsellestriyorum

plt.scatter(kmeans.cluster_centers_[:,   0], kmeans.cluster_centers_[:,1],s = 100, c = 'yellow', label = 'Merkez Noktasi')   #Kumelerin merkez noktalarini olusturmasini sagliyorum

print('Dogruluk Orani = ',accuracy_score(y_pred,y_test)) #Dogruluk oranini yazdiriyorum

plt.legend() #Grafik elemanlarini adlandirdigim sekilde grafikte gosterttim
plt.show() #Grafigi ortaya cikarttim


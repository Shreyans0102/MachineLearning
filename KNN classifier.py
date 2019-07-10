from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 1)
knn = KNeighborsClassifier(n_neighbors = 2)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

print("Knn model accuracy:",metrics.accuracy_score(y_test,y_pred))

sample = [[3,4,5,2],[2,3,5,4]]

preds= knn.predict(sample)
pred_species = [iris.target_names[p] for p in preds] 
print("Predictions:", pred_species) 




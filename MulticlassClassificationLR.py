import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#LogisticRegression(solver='lbfgs')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)



digits = load_digits()

print(dir(digits))

#print(digits.data[0])#it gives numeric data of image
#print(load_digits.__doc__)
##plt.gray()
##plt.matshow(d.images[1796])
##plt.show()#it shows actual image
##for i in range(5):
##    plt.gray()
##    plt.matshow(digits.images[i])
##    plt.show()
#print(digits.target[0:5])
X_train,x_test,y_train,y_test = train_test_split(digits.data,digits.target,test_size=0.2)
model=LogisticRegression(penalty='l1',dual=False,max_iter=110, solver='liblinear')
model.fit(X_train,y_train)
print(model.score(x_test,y_test))
##plt.matshow(digits.images[67])
##plt.show()
print(digits.target[1397])
print(model.predict([digits.data[90]]))
#predict method require multidimensional array
for i in range(100,200,5):
    print(model.predict([digits.data[i]]))

y_pred=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
plt.show()


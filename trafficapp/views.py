# Libraries
from django.shortcuts import render
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


df=pd.read_csv('static/Kaagle_Upload.csv')

df=df.dropna()

df.head()


df2 = df[['special_conditions_at_site','pedestrian_movement','road_surface_conditions','light_conditions','weather_conditions','age_of_vehicle','sex_of_driver','age_of_driver','junction_location', 'junction_detail','junction_control','did_police_officer_attend_scene_of_accident','accident_severity','day_of_week']]


df2.replace(-1, np.nan, inplace=True) # Same as previously
df2=df2.dropna()


df2 =df2.drop(['did_police_officer_attend_scene_of_accident'], axis = 1)

corrmat = df2.corr()
f, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corrmat, vmax=.8, square=True)

########## converting the columns into normalised form ########
df2['age_of_driver'] = np.log1p(df2['age_of_driver'])
df2['age_of_vehicle'] = np.log1p(df2['age_of_vehicle'])# standardise the feature

########### Here we can iclude the weather or exclude the weather as well ########
################ but e are including the weather as well ###########
#df1= df1[:15000] #keep 15000 to decrease running times

df2= df2[:15000] #keep 15000
Y = df2.accident_severity.values
df2 =df2.drop(['accident_severity'],axis = 1)
X = df2.iloc[:,:].values

X_train, X_test,Y_train,Y_test = train_test_split(X, Y, test_size=0.33, random_state=99)


################ Home #################
def home(request):
    return render(request,'index.html')


######## SVM ######
def svm(request):
    svc = SVC()
    svc.fit(X_train, Y_train)
    Y_pred = svc.predict(X_test)
    acc_svc = round(svc.score(X_test, Y_test) * 100, 2)
    cm = confusion_matrix(Y_test, Y_pred)
    d = {'a': acc_svc, 'b': cm}
    return render(request,'svm.html',d)


###### KNN#######
def knn(request):
    knn = KNeighborsClassifier(n_neighbors=10)
    knn.fit(X_train, Y_train)
    Y_pred = knn.predict(X_test)
    acc_knn = round(knn.score(X_test, Y_test) * 100, 2)
    cm = confusion_matrix(Y_test, Y_pred)
    d = {'a': acc_knn, 'b': cm}
    return render(request,'knn.html',d)


######Logistic Regression######
def logistic(request):
    logreg = LogisticRegression()
    logreg.fit(X_train, Y_train)
    Y_pred = logreg.predict(X_test)
    acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
    cm = confusion_matrix(Y_test, Y_pred)
    d = {'a': acc_log, 'b': cm}
    return render(request,'lt.html',d)


#####Dicision Tree######
def dt(request):
    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(X_train, Y_train)
    Y_pred = decision_tree.predict(X_test)
    acc_decision_tree = round(decision_tree.score(X_test, Y_test) * 100, 2)
    cm = confusion_matrix(Y_test, Y_pred)
    d = {'a': acc_decision_tree, 'b': cm}
    return render(request,'dt.html',d)
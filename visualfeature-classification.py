import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import Normalizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import mean_squared_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

os.system('clear')


#features1 = pd.read_csv('features/trailer/Visual/AlexNet features/Med/AlexNetFeatures - MED - fc7.csv')
#print(features1.shape)
#print(features1.head())

features1 = pd.read_csv('features/trailer/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')

#features2 = pd.read_csv('features/trailer/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
#print(features2.shape)
#print(features2.head())



#features_merged = featuresA.merge(featuresB, on='movieId',how = 'inner')
#print(features_merged.shape)
#print(features_merged.head())


ratings = pd.read_csv('movieClipsRatingsTrain_AvgStd.csv')
ratings = ratings.groupby('movieId').first()

#print(ratings.shape)
#print(ratings.head())

rating_features_merged = ratings.merge(features1, on='movieId',how = 'inner')
#print(rating_features_merged.shape)
#print(rating_features_merged)


# X: features, y: labels  seperated
X = rating_features_merged.iloc[:, 4:]
print(X.head())
y = rating_features_merged['avgRating']
print(y.head())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

y = round(y*2)/2
y_train = round(y_train*2)/2

le = preprocessing.LabelEncoder()
le.fit(y)
#print(le.classes_)
y_train = le.transform(y_train)

print(y_train)

normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

#Logistic regression classifier
logreg = LogisticRegression()
logreg.fit(X_train,y_train)

y_test_pred = logreg.predict(X_test)
y_test_pred = le.inverse_transform(y_test_pred)

#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)

#y_test_pred = knn.predict(X_test)
#y_test_pred = le.inverse_transform(y_test_pred)

#clf = RandomForestClassifier(n_estimators=100, random_state = 10)
#clf.fit(X_train, y_train)

#y_test_pred = clf.predict(X_test)
#y_test_pred = le.inverse_transform(y_test_pred)



MSE = mean_squared_error(y_test_pred, y_test)
print('RMSE of LogisticRegression classifier on test set: {:.2f}'.format(np.sqrt(MSE))) 




#ratings = ratings.groupby('movieId').first()
#print( (ratings))

#ratings_with_label = pd.DataFrame(ratings.iloc[:, 1])
#print(ratings_with_label.head())

#trailers_merged_rating = trailers.merge(ratings_with_label, on='movieId',how = 'inner')
#print(trailers_merged_rating.head())

# X: features, y: labels  seperated

#X = trailers_merged_rating.iloc[:,1:10228]
#print(X)
#y = trailers_merged_rating['avgRating']
#print(y.head())

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

#y = round(y*2)/2
#y_train = round(y_train*2)/2

#le = preprocessing.LabelEncoder()
#le.fit(y)
#print(le.classes_)
#y_train = le.transform(y_train)

#print(y_train)

#normalizer = Normalizer()
#X_train = normalizer.fit_transform(X_train)
#X_test = normalizer.transform(X_test)

#logreg = LogisticRegression()
#logreg.fit(X_train,y_train)

#y_test_pred = logreg.predict(X_test)
#y_test_pred = le.inverse_transform(y_test_pred)

#knn = KNeighborsClassifier()
#knn.fit(X_train, y_train)

#y_test_pred = knn.predict(X_test)
#y_test_pred = le.inverse_transform(y_test_pred)

#clf = RandomForestClassifier(n_estimators=100, random_state = 10)
#clf.fit(X_train, y_train)

#y_test_pred = clf.predict(X_test)
#y_test_pred = le.inverse_transform(y_test_pred)


#MSE = mean_squared_error(y_test_pred, y_test)
#print('RMSE of RandomForestClassifier classifier on test set: {:.2f}'.format(np.sqrt(MSE))) 















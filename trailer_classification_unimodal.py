import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
from sklearn.impute import SimpleImputer


os.system('clear')

feat_folder = '/Users/MyHome/Documents/GitHub/MediaEvalFeatures/features/trailer'
rating_folder = '/Users/MyHome/Documents/GitHub/MediaEvalFeatures'


feature_name = 'AVF(AVGVAR)'    # ivector(256,100), BLF, Deep(AVG), AVF, tag, genre
classifer_type = 'Knn'       # LogisticRegression, Knn,  RandomForestClassifier



if feature_name == 'ivector(256,100)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_100.csv')
elif feature_name == 'ivector(256,200)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_200.csv')
elif feature_name == 'ivector(256,400)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_256_tvDim_400.csv')
elif feature_name == 'ivector(512,100)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_100.csv')
elif feature_name == 'ivector(512,200)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_200.csv')
elif feature_name == 'ivector(512,400)':
	features = pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_512_tvDim_400.csv')


elif feature_name == 'BLF':
	features2_1 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_CORRELATIONfeat.csv')
	features2_2 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_DELTASPECTRALfeat.csv')
	features2_3 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_LOGARITHMICFLUCTUATIONfeat.csv')
	features2_4 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_SPECTRALCONTRASTfeat.csv')
	features2_5 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_SPECTRALfeat.csv')
	features2_6 = pd.read_csv(feat_folder + '/Audio/Block level features/Component6/BLF_VARIANCEDELTASPECTRALfeat.csv')

	features_merged = features2_1.merge(features2_2, on='movieId',how = 'inner')
	#print(features_merged.shape)
	features_merged = features_merged.merge(features2_3, on='movieId',how = 'inner')
	#print(features_merged.shape)
	features_merged = features_merged.merge(features2_4, on='movieId',how = 'inner')
	#print(features_merged.shape)
	features_merged = features_merged.merge(features2_5, on='movieId',how = 'inner')
	#print(features_merged.shape)
	features = features_merged.merge(features2_6, on='movieId',how = 'inner')
	#print(features_merged.shape)
elif feature_name == 'Deep(AVG)':
	features = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')

elif feature_name == 'Deep(MED)':
	features = pd.read_csv(feat_folder + '/Visual/AlexNet features/Med/AlexNetFeatures - MED - fc7.csv')

elif feature_name == 'Deep(AVGVAR)':
	features = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')

elif feature_name == 'Deep(MEDMAD)':
	features = pd.read_csv(feat_folder + '/Visual/AlexNet features/MedMad/AlexNetFeatures - MEDMAD - fc7.csv')

elif feature_name == 'AVF(AVG)':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)

	print(np.any(np.isnan(features))) #and gets False
	print(np.all(np.isfinite(features))) #and gets True

elif feature_name == 'AVF(AVGVAR)':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/AvgVar/AestheticFeatures - AVGVAR - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)

	print(np.any(np.isnan(features))) #and gets False
	print(np.all(np.isfinite(features))) #and gets True

elif feature_name == 'AVF(MED)':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/MedMad/AestheticFeatures - Med - All.csv')


elif feature_name == 'AVF(MEDMAD)':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Med/AestheticFeatures - Med - All.csv')

	#features = features.iloc[:,0:108]    # Only to get AVG from AVGVAR
	#print(features3.shape)
	#print(features3.head())

elif feature_name == 'Tag':
	features = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')

elif feature_name == 'Genre':
	features = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')




#print(features_merged.shape)
#print(features_merged.head())



ratings = pd.read_csv(rating_folder + '/movieClipsRatingsTrain_AvgStd.csv')
ratings = ratings.groupby('movieId').first()

#print(ratings.shape)
#print(ratings.head())

rating_features_merged = ratings.merge(features, on='movieId',how = 'inner')
print(rating_features_merged.shape)
print(rating_features_merged.head())
#print(list(rating_features_merged))


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
#print(y_train)
y_train = le.transform(y_train)

# print(X_train)
# print(X_test)

print(np.any(np.isnan(X_train))) #and gets False
print(np.all(np.isfinite(X_train))) #and gets True

normalizer = Normalizer()
X_train = normalizer.fit_transform(X_train)
X_test = normalizer.transform(X_test)

if classifer_type == 'LogisticRegression':
#Logistic regression classifier
	logreg = LogisticRegression()
	logreg.fit(X_train,y_train)

	y_test_pred = logreg.predict(X_test)
#print(y_test_pred)
	y_test_pred = le.inverse_transform(y_test_pred)

elif classifer_type == 'Knn':
	knn = KNeighborsClassifier()
	knn.fit(X_train, y_train)

	y_test_pred = knn.predict(X_test)
	y_test_pred = le.inverse_transform(y_test_pred)

elif classifer_type == 'RandomForestClassifier':
	clf = RandomForestClassifier(n_estimators=100, random_state = 10)
	clf.fit(X_train, y_train)

	y_test_pred = clf.predict(X_test)
	y_test_pred = le.inverse_transform(y_test_pred)



print('----------------------------------------------- \n') 
MSE = mean_squared_error(y_test_pred, y_test)
print('RMSE of ' + classifer_type + ' classifier for ' + feature_name + ' on test set: {:.2f} \n'.format(np.sqrt(MSE))) 
print('----------------------------------------------- \n') 





















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















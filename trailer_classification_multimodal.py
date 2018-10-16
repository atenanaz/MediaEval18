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


feature_name = 'Deep(AVGVAR) + AVF(AVG)'    # ivector(256,100), BLF, Deep(AVG), AVF, tag, genre
classifer_type = 'RandomForestClassifier'       # LogisticRegression, Knn,  RandomForestClassifier


ivec_gmm = '256'
tvDim = '100'

if feature_name == 'ivector + BLF':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')


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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')


	features_merged = features2.merge(features1, on='movieId',how = 'inner')

	print(features1.shape)
	print(features2.shape)
	print(features_merged.shape)

elif feature_name == 'ivector + Deep(AVG)':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')
	features3 = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')
	features_merged = features3.merge(features1, on='movieId',how = 'inner')

	print(features1.shape)
	print(features3.shape)
	print(features_merged.shape)

elif feature_name == 'ivector + Deep(AVGVAR)':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')
	features3 = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')
	features_merged = features3.merge(features1, on='movieId',how = 'inner')

	print(features1.shape)
	print(features3.shape)
	print(features_merged.shape)

elif feature_name == 'ivector + AVF(AVG)':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)
	features_merged = features.merge(features1, on='movieId',how = 'inner')

	#print(np.any(np.isnan(features))) #and gets False
	#print(np.all(np.isfinite(features))) #and gets True

	print(features1.shape)
	print(features.shape)
	print(features_merged.shape)

elif feature_name == 'ivector + Tag':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')
	features5 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features_merged = features5.merge(features1, on='movieId',how = 'inner')

	print(features1.shape)
	print(features5.shape)
	print(features_merged.shape)

elif feature_name == 'ivector + Genre':
	features1= pd.read_csv(feat_folder + '/Audio/ivector features/IVec_splitItem_fold_1_gmm_' + ivec_gmm +'_tvDim_'+ tvDim +'.csv')
	features6 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features6.merge(features1, on='movieId',how = 'inner')

elif feature_name == 'BLF + Deep(AVG)':
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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')


	features3 = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')
	features_merged = features3.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'BLF + Deep(AVGVAR)':
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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')


	features3 = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')
	features_merged = features3.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'BLF + AVF(AVG)':
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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')

	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)

	features_merged = features.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'BLF + Tag':
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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')
	features5 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features_merged = features5.merge(features2, on='movieId',how = 'inner')


elif feature_name == 'BLF + Genre':
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
	features2 = features_merged.merge(features2_6, on='movieId',how = 'inner')
	features6 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features6.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVG) + AVF(AVG)':
	features1 = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')
	

	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)
	features_merged = features1.merge(features, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVGVAR) + AVF(AVG)':
	features1 = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')

	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)
	features_merged = features.merge(features1, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVG) + Tag':
	features2 = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')
	features3 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features_merged = features3.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVGVAR) + Tag':
	features2 = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')
	features3 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features_merged = features3.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVG) + Genre':
	features2 = pd.read_csv(feat_folder + '/Visual/AlexNet features/Avg/AlexNetFeatures - AVG - fc7.csv')
	features4 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features4.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'Deep(AVGVAR) + Genre':
	features2 = pd.read_csv(feat_folder + '/Visual/AlexNet features/AvgVar/AlexNetFeatures - AVGVAR - fc7.csv')
	features4 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features4.merge(features2, on='movieId',how = 'inner')

elif feature_name == 'AVF(AVG) + Tag':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)
	features5 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features_merged = features5.merge(features, on='movieId',how = 'inner')

elif feature_name == 'AVF(AVG) + Genre':
	features = pd.read_csv(feat_folder + '/Visual/Aesthetic features/Avg/AestheticFeatures - AVG - All.csv')
	columns = features.columns
	features = features.replace([np.inf, -np.inf], np.nan)
	imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')
	imp_mean.fit(features)
	features = imp_mean.transform(features)
	features = pd.DataFrame(features, columns=columns)
	features5 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features5.merge(features, on='movieId',how = 'inner')

elif feature_name == 'Tag + Genre':
	features1 = pd.read_csv(feat_folder + '/Metadata/TagFeatures.csv')
	features2 = pd.read_csv(feat_folder + '/Metadata/GenreFeatures.csv')
	features_merged = features2.merge(features1, on='movieId',how = 'inner')


ratings = pd.read_csv(rating_folder + '/movieClipsRatingsTrain_AvgStd.csv')
ratings = ratings.groupby('movieId').first()

#print(ratings.shape)
#print(ratings.head())

rating_features_merged = ratings.merge(features_merged, on='movieId',how = 'inner')
print(rating_features_merged.shape)
#print(rating_features_merged.head())
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















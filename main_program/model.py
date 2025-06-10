import pickle, os
import pandas as pd
import numpy as np
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
import shapelet
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
with open("mfcc_features.pkl","rb") as f:
    mfcc_features=pickle.load(f) 
project_root=os.path.dirname(os.getcwd())
audio_mfcc_feature_20s_2048_512=mfcc_features["audio_mfcc_feature_20s_2048_512"]
audio_mfcc_feature_20s_2048_256=mfcc_features["audio_mfcc_feature_20s_2048_256"]
audio_mfcc_feature_20s_1024_256=mfcc_features["audio_mfcc_feature_20s_1024_256"]
audio_mfcc_feature_20s_1024_128=mfcc_features["audio_mfcc_feature_20s_1024_128"]

audio_mfcc_feature_50s_2048_512=mfcc_features["audio_mfcc_feature_50s_2048_512"]
audio_mfcc_feature_50s_2048_256=mfcc_features["audio_mfcc_feature_50s_2048_256"]
audio_mfcc_feature_50s_1024_256=mfcc_features["audio_mfcc_feature_50s_1024_256"]
audio_mfcc_feature_50s_1024_128=mfcc_features["audio_mfcc_feature_50s_1024_128"]

lab=[]
for i in list(range(4)):
    lab+=[str(i)]*100
label=pd.DataFrame(lab)

total_samples=400
indices=np.arange(total_samples)
train_idx, temp_idx=train_test_split(indices, test_size=0.5, random_state=123)
val_idx, test_idx=train_test_split(temp_idx, test_size=0.5, random_state=123)

train_label=[lab[i] for i in train_idx]
val_label=[lab[i] for i in val_idx]
test_label=[lab[i] for i in test_idx]
###########################Data sets splitting############################

train_mfcc_data_2048_512=audio_mfcc_feature_20s_2048_512.iloc[train_idx]
val_mfcc_data_2048_512=audio_mfcc_feature_20s_2048_512.iloc[val_idx]
test_mfcc_data_2048_512=audio_mfcc_feature_20s_2048_512.iloc[test_idx]

train_mfcc_data_2048_256=audio_mfcc_feature_20s_2048_256.iloc[train_idx]
val_mfcc_data_2048_256=audio_mfcc_feature_20s_2048_256.iloc[val_idx]
test_mfcc_data_2048_256=audio_mfcc_feature_20s_2048_256.iloc[test_idx]

train_mfcc_data_1024_256=audio_mfcc_feature_20s_1024_256.iloc[train_idx]
val_mfcc_data_1024_256=audio_mfcc_feature_20s_1024_256.iloc[val_idx]
test_mfcc_data_1024_256=audio_mfcc_feature_20s_1024_256.iloc[test_idx]

train_mfcc_data_1024_128=audio_mfcc_feature_20s_1024_128.iloc[train_idx]
val_mfcc_data_1024_128=audio_mfcc_feature_20s_1024_128.iloc[val_idx]
test_mfcc_data_1024_128=audio_mfcc_feature_20s_1024_128.iloc[test_idx]

mfcc0_mean_2048_512=audio_mfcc_feature_50s_2048_512.filter(regex='mean_0', axis=1)
mfccmean0_2048_512=[row.to_numpy() for _,row in mfcc0_mean_2048_512.iterrows()]

mfcc0_mean_2048_256=audio_mfcc_feature_50s_2048_256.filter(regex='mean_0', axis=1)
mfccmean0_2048_256=[row.to_numpy() for _,row in mfcc0_mean_2048_256.iterrows()]

mfcc0_mean_1024_256=audio_mfcc_feature_50s_1024_256.filter(regex='mean_0', axis=1)
mfccmean0_1024_256=[row.to_numpy() for _,row in mfcc0_mean_1024_256.iterrows()]

mfcc0_mean_1024_128=audio_mfcc_feature_50s_1024_128.filter(regex='mean_0', axis=1)
mfccmean0_1024_128=[row.to_numpy() for _,row in mfcc0_mean_1024_128.iterrows()]

train_mfcc0_2048_512=[mfccmean0_2048_512[i] for i in train_idx]
val_mfcc0_2048_512=[mfccmean0_2048_512[i] for i in val_idx]
test_mfcc0_2048_512=[mfccmean0_2048_512[i] for i in test_idx]

train_mfcc0_2048_256=[mfccmean0_2048_256[i] for i in train_idx]
val_mfcc0_2048_256=[mfccmean0_2048_256[i] for i in val_idx]
test_mfcc0_2048_256=[mfccmean0_2048_256[i] for i in test_idx]

train_mfcc0_1024_256=[mfccmean0_1024_256[i] for i in train_idx]
val_mfcc0_1024_256=[mfccmean0_1024_256[i] for i in val_idx]
test_mfcc0_1024_256=[mfccmean0_1024_256[i] for i in test_idx]

train_mfcc0_1024_128=[mfccmean0_1024_128[i] for i in train_idx]
val_mfcc0_1024_128=[mfccmean0_1024_128[i] for i in val_idx]
test_mfcc0_1024_128=[mfccmean0_1024_128[i] for i in test_idx]

#################################Statistical Models####################################

############################ with n_fft=2048 and hop_length=512 #############################
scaler=StandardScaler()
std_train_mfcc_data_2048_512=scaler.fit_transform(train_mfcc_data_2048_512)
std_val_mfcc_data_2048_512=scaler.fit_transform(val_mfcc_data_2048_512)
std_test_mfcc_data_2048_512=scaler.fit_transform(test_mfcc_data_2048_512)
## KNN with cosine distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i,metric="cosine")
    knn.fit(train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_knn_2048_512=knn.predict(val_mfcc_data_2048_512)
    val_acc_knn_2048_512=accuracy_score(val_label,mfcc_val_pred_knn_2048_512)
    if val_acc_knn_2048_512>acc:
        acc=val_acc_knn_2048_512
        best_n=i
knn_cosine_2048_512=KNeighborsClassifier(n_neighbors=best_n,metric="cosine")
knn_cosine_2048_512.fit(train_mfcc_data_2048_512,train_label)
test_pred_knn_cosine_2048_512=knn_cosine_2048_512.predict(test_mfcc_data_2048_512)
test_acc_knn_cosine_2048_512=accuracy_score(test_label,test_pred_knn_cosine_2048_512)
test_cr_knn_cosine_2048_512=classification_report(test_label,test_pred_knn_cosine_2048_512)
test_cm_knn_cosine_2048_512=confusion_matrix(test_label,test_pred_knn_cosine_2048_512)

## KNN with Euclidean distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_knn_2048_512=knn.predict(val_mfcc_data_2048_512)
    val_acc_knn_2048_512=accuracy_score(val_label,mfcc_val_pred_knn_2048_512)
    if val_acc_knn_2048_512>acc:
        acc=val_acc_knn_2048_512
        best_n=i
knn_euclidean_2048_512=KNeighborsClassifier(n_neighbors=best_n)
knn_euclidean_2048_512.fit(train_mfcc_data_2048_512,train_label)
test_pred_knn_euclidean_2048_512=knn_euclidean_2048_512.predict(test_mfcc_data_2048_512)
test_acc_knn_euclidean_2048_512=accuracy_score(test_label,test_pred_knn_euclidean_2048_512)
test_cr_knn_euclidean_2048_512=classification_report(test_label,test_pred_knn_euclidean_2048_512)
test_cm_knn_euclidean_2048_512=confusion_matrix(test_label,test_pred_knn_euclidean_2048_512)

## KNN with Manhatten distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_knn_2048_512=knn.predict(val_mfcc_data_2048_512)
    val_acc_knn_2048_512=accuracy_score(val_label,mfcc_val_pred_knn_2048_512)
    if val_acc_knn_2048_512>acc:
        acc=val_acc_knn_2048_512
        best_n=i
knn_L1_2048_512=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_L1_2048_512.fit(train_mfcc_data_2048_512,train_label)
test_pred_knn_L1_2048_512=knn_L1_2048_512.predict(test_mfcc_data_2048_512)
test_acc_knn_L1_2048_512=accuracy_score(test_label,test_pred_knn_L1_2048_512)
test_cr_knn_L1_2048_512=classification_report(test_label,test_pred_knn_L1_2048_512)
test_cm_knn_L1_2048_512=confusion_matrix(test_label,test_pred_knn_L1_2048_512)

## KNN with Euclidean distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(std_train_mfcc_data_2048_512,train_label)
    std_mfcc_val_pred_knn_2048_512=knn.predict(std_val_mfcc_data_2048_512)
    std_val_acc_knn_2048_512=accuracy_score(val_label,std_mfcc_val_pred_knn_2048_512)
    if std_val_acc_knn_2048_512>acc:
        acc=std_val_acc_knn_2048_512
        best_n=i
knn_std_euclidean_2048_512=KNeighborsClassifier(n_neighbors=best_n)
knn_std_euclidean_2048_512.fit(std_train_mfcc_data_2048_512,train_label)
test_pred_knn_std_euclidean_2048_512=knn_std_euclidean_2048_512.predict(std_test_mfcc_data_2048_512)
test_acc_knn_std_euclidean_2048_512=accuracy_score(test_label,test_pred_knn_std_euclidean_2048_512)
test_cr_knn_std_euclidean_2048_512=classification_report(test_label,test_pred_knn_std_euclidean_2048_512)
test_cm_knn_std_euclidean_2048_512=confusion_matrix(test_label,test_pred_knn_std_euclidean_2048_512)

## KNN with Manhatten distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(std_train_mfcc_data_2048_512,train_label)
    std_mfcc_val_pred_knn_2048_512=knn.predict(std_val_mfcc_data_2048_512)
    std_val_acc_knn_2048_512=accuracy_score(val_label,std_mfcc_val_pred_knn_2048_512)
    if std_val_acc_knn_2048_512>acc:
        acc=std_val_acc_knn_2048_512
        best_n=i
knn_std_L1_2048_512=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_std_L1_2048_512.fit(std_train_mfcc_data_2048_512,train_label)
test_pred_knn_std_L1_2048_512=knn_std_L1_2048_512.predict(std_test_mfcc_data_2048_512)
test_acc_knn_std_L1_2048_512=accuracy_score(test_label,test_pred_knn_std_L1_2048_512)
test_cr_knn_std_L1_2048_512=classification_report(test_label,test_pred_knn_std_L1_2048_512)
test_cm_knn_std_L1_2048_512=confusion_matrix(test_label,test_pred_knn_std_L1_2048_512)

## Linear Support Vector Classifier with L1 penalty (Fail to converge)
svml=LinearSVC(penalty="l1",dual=False,C=0.5,max_iter=50000,tol=1e-4)
svml.fit(train_mfcc_data_2048_512,train_label)

## Linear Support Vector Classifier with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10]:
    svml_std=LinearSVC(penalty="l1",dual=False,C=C,max_iter=20000,tol=1e-4,random_state=233)
    svml_std.fit(std_train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_svml1std=svml_std.predict(std_val_mfcc_data_2048_512)
    mfcc_val_acc_svml1std=accuracy_score(val_label,mfcc_val_pred_svml1std)
    if mfcc_val_acc_svml1std>acc:
        acc=mfcc_val_acc_svml1std
        best_C=C
svml_std_2048_512=LinearSVC(penalty="l1",dual=False,C=best_C,max_iter=20000,tol=1e-4)
svml_std_2048_512.fit(std_train_mfcc_data_2048_512,train_label)
test_pred_svml_std_2048_512=svml_std_2048_512.predict(std_test_mfcc_data_2048_512)
test_acc_svml_std_2048_512=accuracy_score(test_label,test_pred_svml_std_2048_512)
test_cr_svml_std_2048_512=classification_report(test_label,test_pred_svml_std_2048_512)
test_cm_svml_std_2048_512=confusion_matrix(test_label,test_pred_svml_std_2048_512)
svm_std_2048_512_coef=svml_std_2048_512.coef_

## Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_LR=LR.predict(val_mfcc_data_2048_512)
    mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if mfcc_val_acc_LR>acc:
        acc=mfcc_val_acc_LR
        best_C=C
LR_2048_512=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_2048_512.fit(train_mfcc_data_2048_512,train_label)
test_pred_LR_2048_512=LR_2048_512.predict(test_mfcc_data_2048_512)
test_acc_LR_2048_512=accuracy_score(test_label,test_pred_LR_2048_512)
test_cr_LR_2048_512=classification_report(test_label,test_pred_LR_2048_512)
test_cm_LR_2048_512=confusion_matrix(test_label,test_pred_LR_2048_512)
LR_2048_512_coef=LR_2048_512.coef_
zero_per_LR_2048_512=np.sum(LR_2048_512_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(train_mfcc_data_2048_512,train_label)
    mfcc_val_pred_MLR=MLR.predict(val_mfcc_data_2048_512)
    mfcc_val_acc_MLR=accuracy_score(val_label,mfcc_val_pred_MLR)
    if mfcc_val_acc_MLR>acc:
        acc=mfcc_val_acc_MLR
        best_C=C
MLR_2048_512=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_2048_512.fit(train_mfcc_data_2048_512,train_label)
test_pred_MLR_2048_512=MLR_2048_512.predict(test_mfcc_data_2048_512)
test_acc_MLR_2048_512=accuracy_score(test_label,test_pred_MLR_2048_512)
test_cr_MLR_2048_512=classification_report(test_label,test_pred_MLR_2048_512)
test_cm_MLR_2048_512=confusion_matrix(test_label,test_pred_MLR_2048_512)
MLR_2048_512_coef=MLR_2048_512.coef_
zero_per_MLR_2048_512=np.sum(MLR_2048_512_coef==0,axis=1)

## Logistic Regression with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(std_train_mfcc_data_2048_512,train_label)
    std_mfcc_val_pred_LR=LR.predict(std_val_mfcc_data_2048_512)
    std_mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if std_mfcc_val_acc_LR>acc:
        acc=std_mfcc_val_acc_LR
        best_C=C
LR_std_2048_512=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_std_2048_512.fit(std_train_mfcc_data_2048_512,train_label)
std_test_pred_LR_2048_512=LR_2048_512.predict(std_test_mfcc_data_2048_512)
std_test_acc_LR_2048_512=accuracy_score(test_label,std_test_pred_LR_2048_512)
std_test_cr_LR_2048_512=classification_report(test_label,std_test_pred_LR_2048_512)
std_test_cm_LR_2048_512=confusion_matrix(test_label,std_test_pred_LR_2048_512)
std_LR_2048_512_coef=LR_std_2048_512.coef_
zero_per_LR_std_2048_512=np.sum(std_LR_2048_512_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty with standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(std_train_mfcc_data_2048_512,train_label)
    std_mfcc_val_pred_MLR=MLR.predict(std_val_mfcc_data_2048_512)
    std_mfcc_val_acc_MLR=accuracy_score(val_label,std_mfcc_val_pred_MLR)
    if std_mfcc_val_acc_MLR>acc:
        acc=std_mfcc_val_acc_MLR
        best_C=C
MLR_std_2048_512=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_std_2048_512.fit(std_train_mfcc_data_2048_512,train_label)
std_test_pred_MLR_2048_512=MLR_std_2048_512.predict(std_test_mfcc_data_2048_512)
std_test_acc_MLR_2048_512=accuracy_score(test_label,std_test_pred_MLR_2048_512)
std_test_cr_MLR_2048_512=classification_report(test_label,std_test_pred_MLR_2048_512)
std_test_cm_MLR_2048_512=confusion_matrix(test_label,std_test_pred_MLR_2048_512)
std_MLR_2048_512_coef=MLR_std_2048_512.coef_
zero_per_MLR_std_2048_512=np.sum(std_MLR_2048_512_coef==0,axis=1)


############################ with n_fft=2048 and hop_length=256 #############################
std_train_mfcc_data_2048_256=scaler.fit_transform(train_mfcc_data_2048_256)
std_val_mfcc_data_2048_256=scaler.fit_transform(val_mfcc_data_2048_256)
std_test_mfcc_data_2048_256=scaler.fit_transform(test_mfcc_data_2048_256)
## KNN with cosine distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i,metric="cosine")
    knn.fit(train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_knn_2048_256=knn.predict(val_mfcc_data_2048_256)
    val_acc_knn_2048_256=accuracy_score(val_label,mfcc_val_pred_knn_2048_256)
    if val_acc_knn_2048_256>acc:
        acc=val_acc_knn_2048_256
        best_n=i
knn_cosine_2048_256=KNeighborsClassifier(n_neighbors=best_n,metric="cosine")
knn_cosine_2048_256.fit(train_mfcc_data_2048_256,train_label)
test_pred_knn_cosine_2048_256=knn_cosine_2048_256.predict(test_mfcc_data_2048_256)
test_acc_knn_cosine_2048_256=accuracy_score(test_label,test_pred_knn_cosine_2048_256)
test_cr_knn_cosine_2048_256=classification_report(test_label,test_pred_knn_cosine_2048_256)
test_cm_knn_cosine_2048_256=confusion_matrix(test_label,test_pred_knn_cosine_2048_256)

## KNN with Euclidean distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_knn_2048_256=knn.predict(val_mfcc_data_2048_256)
    val_acc_knn_2048_256=accuracy_score(val_label,mfcc_val_pred_knn_2048_256)
    if val_acc_knn_2048_256>acc:
        acc=val_acc_knn_2048_256
        best_n=i
knn_euclidean_2048_256=KNeighborsClassifier(n_neighbors=best_n)
knn_euclidean_2048_256.fit(train_mfcc_data_2048_256,train_label)
test_pred_knn_euclidean_2048_256=knn_euclidean_2048_256.predict(test_mfcc_data_2048_256)
test_acc_knn_euclidean_2048_256=accuracy_score(test_label,test_pred_knn_euclidean_2048_256)
test_cr_knn_euclidean_2048_256=classification_report(test_label,test_pred_knn_euclidean_2048_256)
test_cm_knn_euclidean_2048_256=confusion_matrix(test_label,test_pred_knn_euclidean_2048_256)

## KNN with Manhatten distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_knn_2048_256=knn.predict(val_mfcc_data_2048_256)
    val_acc_knn_2048_256=accuracy_score(val_label,mfcc_val_pred_knn_2048_256)
    if val_acc_knn_2048_256>acc:
        acc=val_acc_knn_2048_256
        best_n=i
knn_L1_2048_256=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_L1_2048_256.fit(train_mfcc_data_2048_256,train_label)
test_pred_knn_L1_2048_256=knn_L1_2048_256.predict(test_mfcc_data_2048_256)
test_acc_knn_L1_2048_256=accuracy_score(test_label,test_pred_knn_L1_2048_256)
test_cr_knn_L1_2048_256=classification_report(test_label,test_pred_knn_L1_2048_256)
test_cm_knn_L1_2048_256=confusion_matrix(test_label,test_pred_knn_L1_2048_256)

## KNN with Euclidean distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(std_train_mfcc_data_2048_256,train_label)
    std_mfcc_val_pred_knn_2048_256=knn.predict(std_val_mfcc_data_2048_256)
    std_val_acc_knn_2048_256=accuracy_score(val_label,std_mfcc_val_pred_knn_2048_256)
    if std_val_acc_knn_2048_256>acc:
        acc=std_val_acc_knn_2048_256
        best_n=i
knn_std_euclidean_2048_256=KNeighborsClassifier(n_neighbors=best_n)
knn_std_euclidean_2048_256.fit(std_train_mfcc_data_2048_256,train_label)
test_pred_knn_std_euclidean_2048_256=knn_std_euclidean_2048_256.predict(std_test_mfcc_data_2048_256)
test_acc_knn_std_euclidean_2048_256=accuracy_score(test_label,test_pred_knn_std_euclidean_2048_256)
test_cr_knn_std_euclidean_2048_256=classification_report(test_label,test_pred_knn_std_euclidean_2048_256)
test_cm_knn_std_euclidean_2048_256=confusion_matrix(test_label,test_pred_knn_std_euclidean_2048_256)

## KNN with Manhatten distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(std_train_mfcc_data_2048_256,train_label)
    std_mfcc_val_pred_knn_2048_256=knn.predict(std_val_mfcc_data_2048_256)
    std_val_acc_knn_2048_256=accuracy_score(val_label,std_mfcc_val_pred_knn_2048_256)
    if std_val_acc_knn_2048_256>acc:
        acc=std_val_acc_knn_2048_256
        best_n=i
knn_std_L1_2048_256=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_std_L1_2048_256.fit(std_train_mfcc_data_2048_256,train_label)
test_pred_knn_std_L1_2048_256=knn_std_L1_2048_256.predict(std_test_mfcc_data_2048_256)
test_acc_knn_std_L1_2048_256=accuracy_score(test_label,test_pred_knn_std_L1_2048_256)
test_cr_knn_std_L1_2048_256=classification_report(test_label,test_pred_knn_std_L1_2048_256)
test_cm_knn_std_L1_2048_256=confusion_matrix(test_label,test_pred_knn_std_L1_2048_256)

## Linear Support Vector Classifier with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10]:
    svml_std=LinearSVC(penalty="l1",dual=False,C=C,max_iter=20000,tol=1e-4,random_state=233)
    svml_std.fit(std_train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_svml1std=svml_std.predict(std_val_mfcc_data_2048_256)
    mfcc_val_acc_svml1std=accuracy_score(val_label,mfcc_val_pred_svml1std)
    if mfcc_val_acc_svml1std>acc:
        acc=mfcc_val_acc_svml1std
        best_C=C
svml_std_2048_256=LinearSVC(penalty="l1",dual=False,C=best_C,max_iter=20000,tol=1e-4,random_state=233)
svml_std_2048_256.fit(std_train_mfcc_data_2048_256,train_label)
test_pred_svml_std_2048_256=svml_std_2048_256.predict(std_test_mfcc_data_2048_256)
test_acc_svml_std_2048_256=accuracy_score(test_label,test_pred_svml_std_2048_256)
svm_std_2048_256_coef=svml_std_2048_256.coef_

## Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_LR=LR.predict(val_mfcc_data_2048_256)
    mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if mfcc_val_acc_LR>acc:
        acc=mfcc_val_acc_LR
        best_C=C
LR_2048_256=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_2048_256.fit(train_mfcc_data_2048_256,train_label)
test_pred_LR_2048_256=LR_2048_256.predict(test_mfcc_data_2048_256)
test_acc_LR_2048_256=accuracy_score(test_label,test_pred_LR_2048_256)
test_cr_LR_2048_256=classification_report(test_label,test_pred_LR_2048_256)
test_cm_LR_2048_256=confusion_matrix(test_label,test_pred_LR_2048_256)
LR_2048_256_coef=LR_2048_256.coef_
zero_per_LR_2048_256=np.sum(LR_2048_256_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(train_mfcc_data_2048_256,train_label)
    mfcc_val_pred_MLR=MLR.predict(val_mfcc_data_2048_256)
    mfcc_val_acc_MLR=accuracy_score(val_label,mfcc_val_pred_MLR)
    if mfcc_val_acc_MLR>acc:
        acc=mfcc_val_acc_MLR
        best_C=C
MLR_2048_256=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_2048_256.fit(train_mfcc_data_2048_256,train_label)
test_pred_MLR_2048_256=MLR_2048_256.predict(test_mfcc_data_2048_256)
test_acc_MLR_2048_256=accuracy_score(test_label,test_pred_MLR_2048_256)
test_cr_MLR_2048_256=classification_report(test_label,test_pred_MLR_2048_256)
test_cm_MLR_2048_256=confusion_matrix(test_label,test_pred_MLR_2048_256)
MLR_2048_256_coef=MLR_2048_256.coef_
zero_per_MLR_2048_256=np.sum(MLR_2048_256_coef==0,axis=1)

## Logistic Regression with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(std_train_mfcc_data_2048_256,train_label)
    std_mfcc_val_pred_LR=LR.predict(std_val_mfcc_data_2048_256)
    std_mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if std_mfcc_val_acc_LR>acc:
        acc=std_mfcc_val_acc_LR
        best_C=C
LR_std_2048_256=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_std_2048_256.fit(std_train_mfcc_data_2048_256,train_label)
std_test_pred_LR_2048_256=LR_2048_256.predict(std_test_mfcc_data_2048_256)
std_test_acc_LR_2048_256=accuracy_score(test_label,std_test_pred_LR_2048_256)
std_test_cr_LR_2048_256=classification_report(test_label,std_test_pred_LR_2048_256)
std_test_cm_LR_2048_256=confusion_matrix(test_label,std_test_pred_LR_2048_256)
std_LR_2048_256_coef=LR_std_2048_256.coef_
zero_per_LR_std_2048_256=np.sum(std_LR_2048_256_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty with standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(std_train_mfcc_data_2048_256,train_label)
    std_mfcc_val_pred_MLR=MLR.predict(std_val_mfcc_data_2048_256)
    std_mfcc_val_acc_MLR=accuracy_score(val_label,std_mfcc_val_pred_MLR)
    if std_mfcc_val_acc_MLR>acc:
        acc=std_mfcc_val_acc_MLR
        best_C=C
MLR_std_2048_256=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_std_2048_256.fit(std_train_mfcc_data_2048_256,train_label)
std_test_pred_MLR_2048_256=MLR_std_2048_256.predict(std_test_mfcc_data_2048_256)
std_test_acc_MLR_2048_256=accuracy_score(test_label,std_test_pred_MLR_2048_256)
std_test_cr_MLR_2048_256=classification_report(test_label,std_test_pred_MLR_2048_256)
std_test_cm_MLR_2048_256=confusion_matrix(test_label,std_test_pred_MLR_2048_256)
std_MLR_2048_256_coef=MLR_std_2048_256.coef_
zero_per_MLR_std_2048_256=np.sum(std_MLR_2048_256_coef==0,axis=1)


############################ with n_fft=1024 and hop_length=256 #############################
std_train_mfcc_data_1024_256=scaler.fit_transform(train_mfcc_data_1024_256)
std_val_mfcc_data_1024_256=scaler.fit_transform(val_mfcc_data_1024_256)
std_test_mfcc_data_1024_256=scaler.fit_transform(test_mfcc_data_1024_256)
## KNN with cosine distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i,metric="cosine")
    knn.fit(train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_knn_1024_256=knn.predict(val_mfcc_data_1024_256)
    val_acc_knn_1024_256=accuracy_score(val_label,mfcc_val_pred_knn_1024_256)
    if val_acc_knn_1024_256>acc:
        acc=val_acc_knn_1024_256
        best_n=i
knn_cosine_1024_256=KNeighborsClassifier(n_neighbors=best_n,metric="cosine")
knn_cosine_1024_256.fit(train_mfcc_data_1024_256,train_label)
test_pred_knn_cosine_1024_256=knn_cosine_1024_256.predict(test_mfcc_data_1024_256)
test_acc_knn_cosine_1024_256=accuracy_score(test_label,test_pred_knn_cosine_1024_256)
test_cr_knn_cosine_1024_256=classification_report(test_label,test_pred_knn_cosine_1024_256)
test_cm_knn_cosine_1024_256=confusion_matrix(test_label,test_pred_knn_cosine_1024_256)

## KNN with Euclidean distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_knn_1024_256=knn.predict(val_mfcc_data_1024_256)
    val_acc_knn_1024_256=accuracy_score(val_label,mfcc_val_pred_knn_1024_256)
    if val_acc_knn_1024_256>acc:
        acc=val_acc_knn_1024_256
        best_n=i
knn_euclidean_1024_256=KNeighborsClassifier(n_neighbors=best_n)
knn_euclidean_1024_256.fit(train_mfcc_data_1024_256,train_label)
test_pred_knn_euclidean_1024_256=knn_euclidean_1024_256.predict(test_mfcc_data_1024_256)
test_acc_knn_euclidean_1024_256=accuracy_score(test_label,test_pred_knn_euclidean_1024_256)
test_cr_knn_euclidean_1024_256=classification_report(test_label,test_pred_knn_euclidean_1024_256)
test_cm_knn_euclidean_1024_256=confusion_matrix(test_label,test_pred_knn_euclidean_1024_256)

## KNN with Manhatten distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_knn_1024_256=knn.predict(val_mfcc_data_1024_256)
    val_acc_knn_1024_256=accuracy_score(val_label,mfcc_val_pred_knn_1024_256)
    if val_acc_knn_1024_256>acc:
        acc=val_acc_knn_1024_256
        best_n=i
knn_L1_1024_256=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_L1_1024_256.fit(train_mfcc_data_1024_256,train_label)
test_pred_knn_L1_1024_256=knn_L1_1024_256.predict(test_mfcc_data_1024_256)
test_acc_knn_L1_1024_256=accuracy_score(test_label,test_pred_knn_L1_1024_256)
test_cr_knn_L1_1024_256=classification_report(test_label,test_pred_knn_L1_1024_256)
test_cm_knn_L1_1024_256=confusion_matrix(test_label,test_pred_knn_L1_1024_256)

## KNN with Euclidean distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(std_train_mfcc_data_1024_256,train_label)
    std_mfcc_val_pred_knn_1024_256=knn.predict(std_val_mfcc_data_1024_256)
    std_val_acc_knn_1024_256=accuracy_score(val_label,std_mfcc_val_pred_knn_1024_256)
    if std_val_acc_knn_1024_256>acc:
        acc=std_val_acc_knn_1024_256
        best_n=i
knn_std_euclidean_1024_256=KNeighborsClassifier(n_neighbors=best_n)
knn_std_euclidean_1024_256.fit(std_train_mfcc_data_1024_256,train_label)
test_pred_knn_std_euclidean_1024_256=knn_std_euclidean_1024_256.predict(std_test_mfcc_data_1024_256)
test_acc_knn_std_euclidean_1024_256=accuracy_score(test_label,test_pred_knn_std_euclidean_1024_256)
test_cr_knn_std_euclidean_1024_256=classification_report(test_label,test_pred_knn_std_euclidean_1024_256)
test_cm_knn_std_euclidean_1024_256=confusion_matrix(test_label,test_pred_knn_std_euclidean_1024_256)

## KNN with Manhatten distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(std_train_mfcc_data_1024_256,train_label)
    std_mfcc_val_pred_knn_1024_256=knn.predict(std_val_mfcc_data_1024_256)
    std_val_acc_knn_1024_256=accuracy_score(val_label,std_mfcc_val_pred_knn_1024_256)
    if std_val_acc_knn_1024_256>acc:
        acc=std_val_acc_knn_1024_256
        best_n=i
knn_std_L1_1024_256=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_std_L1_1024_256.fit(std_train_mfcc_data_1024_256,train_label)
test_pred_knn_std_L1_1024_256=knn_std_L1_1024_256.predict(std_test_mfcc_data_1024_256)
test_acc_knn_std_L1_1024_256=accuracy_score(test_label,test_pred_knn_std_L1_1024_256)
std_train_mfcc_data_2048_256=scaler.fit_transform(train_mfcc_data_2048_256)
test_cr_knn_std_L1_1024_256=classification_report(test_label,test_pred_knn_std_L1_1024_256)
test_cm_knn_std_L1_1024_256=confusion_matrix(test_label,test_pred_knn_std_L1_1024_256)

## Linear Support Vector Classifier with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10]:
    svml_std=LinearSVC(penalty="l1",dual=False,C=C,max_iter=20000,tol=1e-4,random_state=233)
    svml_std.fit(std_train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_svml1std=svml_std.predict(std_val_mfcc_data_1024_256)
    mfcc_val_acc_svml1std=accuracy_score(val_label,mfcc_val_pred_svml1std)
    if mfcc_val_acc_svml1std>acc:
        acc=mfcc_val_acc_svml1std
        best_C=C
svml_std_1024_256=LinearSVC(penalty="l1",dual=False,C=best_C,max_iter=20000,tol=1e-4,random_state=233)
svml_std_1024_256.fit(std_train_mfcc_data_1024_256,train_label)
test_pred_svml_std_1024_256=svml_std_1024_256.predict(std_test_mfcc_data_1024_256)
test_acc_svml_std_1024_256=accuracy_score(test_label,test_pred_svml_std_1024_256)
svm_std_1024_256_coef=svml_std_1024_256.coef_

## Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10, 20, 25, 30]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_LR=LR.predict(val_mfcc_data_1024_256)
    mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if mfcc_val_acc_LR>acc:
        acc=mfcc_val_acc_LR
        best_C=C
LR_1024_256=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_1024_256.fit(train_mfcc_data_1024_256,train_label)
test_pred_LR_1024_256=LR_1024_256.predict(test_mfcc_data_1024_256)
test_acc_LR_1024_256=accuracy_score(test_label,test_pred_LR_1024_256)
test_cr_LR_1024_256=classification_report(test_label,test_pred_LR_1024_256)
test_cm_LR_1024_256=confusion_matrix(test_label,test_pred_LR_1024_256)
LR_1024_256_coef=LR_1024_256.coef_
zero_per_LR_1024_256=np.sum(LR_1024_256_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(train_mfcc_data_1024_256,train_label)
    mfcc_val_pred_MLR=MLR.predict(val_mfcc_data_1024_256)
    mfcc_val_acc_MLR=accuracy_score(val_label,mfcc_val_pred_MLR)
    if mfcc_val_acc_MLR>acc:
        acc=mfcc_val_acc_MLR
        best_C=C
MLR_1024_256=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_1024_256.fit(train_mfcc_data_1024_256,train_label)
test_pred_MLR_1024_256=MLR_1024_256.predict(test_mfcc_data_1024_256)
test_acc_MLR_1024_256=accuracy_score(test_label,test_pred_MLR_1024_256)
test_cr_MLR_1024_256=classification_report(test_label,test_pred_MLR_1024_256)
test_cm_MLR_1024_256=confusion_matrix(test_label,test_pred_MLR_1024_256)
MLR_1024_256_coef=MLR_1024_256.coef_
zero_per_MLR_1024_256=np.sum(MLR_1024_256_coef==0,axis=1)

## Logistic Regression with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(std_train_mfcc_data_1024_256,train_label)
    std_mfcc_val_pred_LR=LR.predict(std_val_mfcc_data_1024_256)
    std_mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if std_mfcc_val_acc_LR>acc:
        acc=std_mfcc_val_acc_LR
        best_C=C
LR_std_1024_256=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_std_1024_256.fit(std_train_mfcc_data_1024_256,train_label)
std_test_pred_LR_1024_256=LR_1024_256.predict(std_test_mfcc_data_1024_256)
std_test_acc_LR_1024_256=accuracy_score(test_label,std_test_pred_LR_1024_256)
std_test_cr_LR_1024_256=classification_report(test_label,std_test_pred_LR_1024_256)
std_test_cm_LR_1024_256=confusion_matrix(test_label,std_test_pred_LR_1024_256)
std_LR_1024_256_coef=LR_std_1024_256.coef_
zero_per_LR_std_1024_256=np.sum(std_LR_1024_256_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty with standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(std_train_mfcc_data_1024_256,train_label)
    std_mfcc_val_pred_MLR=MLR.predict(std_val_mfcc_data_1024_256)
    std_mfcc_val_acc_MLR=accuracy_score(val_label,std_mfcc_val_pred_MLR)
    if std_mfcc_val_acc_MLR>acc:
        acc=std_mfcc_val_acc_MLR
        best_C=C
MLR_std_1024_256=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_std_1024_256.fit(std_train_mfcc_data_1024_256,train_label)
std_test_pred_MLR_1024_256=MLR_std_1024_256.predict(std_test_mfcc_data_1024_256)
std_test_acc_MLR_1024_256=accuracy_score(test_label,std_test_pred_MLR_1024_256)
std_test_cr_MLR_1024_256=classification_report(test_label,std_test_pred_MLR_1024_256)
std_test_cm_MLR_1024_256=confusion_matrix(test_label,std_test_pred_MLR_1024_256)
std_MLR_1024_256_coef=MLR_std_1024_256.coef_
zero_per_MLR_std_1024_256=np.sum(std_MLR_1024_256_coef==0,axis=1)


############################ with n_fft=1024 and hop_length=128 #############################
std_train_mfcc_data_1024_128=scaler.fit_transform(train_mfcc_data_1024_128)
std_val_mfcc_data_1024_128=scaler.fit_transform(val_mfcc_data_1024_128)
std_test_mfcc_data_1024_128=scaler.fit_transform(test_mfcc_data_1024_128)
## KNN with cosine distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i,metric="cosine")
    knn.fit(train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_knn_1024_128=knn.predict(val_mfcc_data_1024_128)
    val_acc_knn_1024_128=accuracy_score(val_label,mfcc_val_pred_knn_1024_128)
    if val_acc_knn_1024_128>acc:
        acc=val_acc_knn_1024_128
        best_n=i
knn_cosine_1024_128=KNeighborsClassifier(n_neighbors=best_n,metric="cosine")
knn_cosine_1024_128.fit(train_mfcc_data_1024_128,train_label)
test_pred_knn_cosine_1024_128=knn_cosine_1024_128.predict(test_mfcc_data_1024_128)
test_acc_knn_cosine_1024_128=accuracy_score(test_label,test_pred_knn_cosine_1024_128)
test_cr_knn_cosine_1024_128=classification_report(test_label,test_pred_knn_cosine_1024_128)
test_cm_knn_cosine_1024_128=confusion_matrix(test_label,test_pred_knn_cosine_1024_128)

## KNN with Euclidean distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_knn_1024_128=knn.predict(val_mfcc_data_1024_128)
    val_acc_knn_1024_128=accuracy_score(val_label,mfcc_val_pred_knn_1024_128)
    if val_acc_knn_1024_128>acc:
        acc=val_acc_knn_1024_128
        best_n=i
knn_euclidean_1024_128=KNeighborsClassifier(n_neighbors=best_n)
knn_euclidean_1024_128.fit(train_mfcc_data_1024_128,train_label)
test_pred_knn_euclidean_1024_128=knn_euclidean_1024_128.predict(test_mfcc_data_1024_128)
test_acc_knn_euclidean_1024_128=accuracy_score(test_label,test_pred_knn_euclidean_1024_128)
test_cr_knn_euclidean_1024_128=classification_report(test_label,test_pred_knn_euclidean_1024_128)
test_cm_knn_euclidean_1024_128=confusion_matrix(test_label,test_pred_knn_euclidean_1024_128)

## KNN with Manhatten distance
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_knn_1024_128=knn.predict(val_mfcc_data_1024_128)
    val_acc_knn_1024_128=accuracy_score(val_label,mfcc_val_pred_knn_1024_128)
    if val_acc_knn_1024_128>acc:
        acc=val_acc_knn_1024_128
        best_n=i
knn_L1_1024_128=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_L1_1024_128.fit(train_mfcc_data_1024_128,train_label)
test_pred_knn_L1_1024_128=knn_L1_1024_128.predict(test_mfcc_data_1024_128)
test_acc_knn_L1_1024_128=accuracy_score(test_label,test_pred_knn_L1_1024_128)
test_cr_knn_L1_1024_128=classification_report(test_label,test_pred_knn_L1_1024_128)
test_cm_knn_L1_1024_128=confusion_matrix(test_label,test_pred_knn_L1_1024_128)

## KNN with Euclidean distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i)
    knn.fit(std_train_mfcc_data_1024_128,train_label)
    std_mfcc_val_pred_knn_1024_128=knn.predict(std_val_mfcc_data_1024_128)
    std_val_acc_knn_1024_128=accuracy_score(val_label,std_mfcc_val_pred_knn_1024_128)
    if std_val_acc_knn_1024_128>acc:
        acc=std_val_acc_knn_1024_128
        best_n=i
knn_std_euclidean_1024_128=KNeighborsClassifier(n_neighbors=best_n)
knn_std_euclidean_1024_128.fit(std_train_mfcc_data_1024_128,train_label)
test_pred_knn_std_euclidean_1024_128=knn_std_euclidean_1024_128.predict(std_test_mfcc_data_1024_128)
test_acc_knn_std_euclidean_1024_128=accuracy_score(test_label,test_pred_knn_std_euclidean_1024_128)
test_cr_knn_std_euclidean_1024_128=classification_report(test_label,test_pred_knn_std_euclidean_1024_128)
test_cm_knn_std_euclidean_1024_128=confusion_matrix(test_label,test_pred_knn_std_euclidean_1024_128)

## KNN with Manhatten distance using standard data
acc=0
for i in range(2,10,1):
    knn=KNeighborsClassifier(n_neighbors=i, metric="cityblock")
    knn.fit(std_train_mfcc_data_1024_128,train_label)
    std_mfcc_val_pred_knn_1024_128=knn.predict(std_val_mfcc_data_1024_128)
    std_val_acc_knn_1024_128=accuracy_score(val_label,std_mfcc_val_pred_knn_1024_128)
    if std_val_acc_knn_1024_128>acc:
        acc=std_val_acc_knn_1024_128
        best_n=i
knn_std_L1_1024_128=KNeighborsClassifier(n_neighbors=best_n, metric="cityblock")
knn_std_L1_1024_128.fit(std_train_mfcc_data_1024_128,train_label)
test_pred_knn_std_L1_1024_128=knn_std_L1_1024_128.predict(std_test_mfcc_data_1024_128)
test_acc_knn_std_L1_1024_128=accuracy_score(test_label,test_pred_knn_std_L1_1024_128)
test_cr_knn_std_L1_1024_128=classification_report(test_label,test_pred_knn_std_L1_1024_128)
test_cm_knn_std_L1_1024_128=confusion_matrix(test_label,test_pred_knn_std_L1_1024_128)

## Linear Support Vector Classifier with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10]:
    svml_std=LinearSVC(penalty="l1",dual=False,C=C,max_iter=20000,tol=1e-4,random_state=233)
    svml_std.fit(std_train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_svml1std=svml_std.predict(std_val_mfcc_data_1024_128)
    mfcc_val_acc_svml1std=accuracy_score(val_label,mfcc_val_pred_svml1std)
    if mfcc_val_acc_svml1std>acc:
        acc=mfcc_val_acc_svml1std
        best_C=C
svml_std_1024_128=LinearSVC(penalty="l1",dual=False,C=best_C,max_iter=20000,tol=1e-4,random_state=233)
svml_std_1024_128.fit(std_train_mfcc_data_1024_128,train_label)
test_pred_svml_std_1024_128=svml_std_1024_128.predict(std_test_mfcc_data_1024_128)
test_acc_svml_std_1024_128=accuracy_score(test_label,test_pred_svml_std_1024_128)
svm_std_1024_128_coef=svml_std_1024_128.coef_

## Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_LR=LR.predict(val_mfcc_data_1024_128)
    mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if mfcc_val_acc_LR>acc:
        acc=mfcc_val_acc_LR
        best_C=C
LR_1024_128=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_1024_128.fit(train_mfcc_data_1024_128,train_label)
test_pred_LR_1024_128=LR_1024_128.predict(test_mfcc_data_1024_128)
test_acc_LR_1024_128=accuracy_score(test_label,test_pred_LR_1024_128)
test_cr_LR_1024_128=classification_report(test_label,test_pred_LR_1024_128)
test_cm_LR_1024_128=confusion_matrix(test_label,test_pred_LR_1024_128)
LR_1024_128_coef=LR_1024_128.coef_
zero_per_LR_1024_128=np.sum(LR_1024_128_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(train_mfcc_data_1024_128,train_label)
    mfcc_val_pred_MLR=MLR.predict(val_mfcc_data_1024_128)
    mfcc_val_acc_MLR=accuracy_score(val_label,mfcc_val_pred_MLR)
    if mfcc_val_acc_MLR>acc:
        acc=mfcc_val_acc_MLR
        best_C=C
MLR_1024_128=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_1024_128.fit(train_mfcc_data_1024_128,train_label)
test_pred_MLR_1024_128=MLR_1024_128.predict(test_mfcc_data_1024_128)
test_acc_MLR_1024_128=accuracy_score(test_label,test_pred_MLR_1024_128)
test_cr_LR_1024_128=classification_report(test_label,test_pred_LR_1024_128)
test_cm_LR_1024_128=confusion_matrix(test_label,test_pred_LR_1024_128)
MLR_1024_128_coef=MLR_1024_128.coef_
zero_per_MLR_1024_128=np.sum(MLR_1024_128_coef==0,axis=1)

## Logistic Regression with L1 penalty using standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    LR=LogisticRegression(penalty="l1", C=C, solver="liblinear",random_state=233)
    LR.fit(std_train_mfcc_data_1024_128,train_label)
    std_mfcc_val_pred_LR=LR.predict(std_val_mfcc_data_1024_128)
    std_mfcc_val_acc_LR=accuracy_score(val_label,mfcc_val_pred_LR)
    if std_mfcc_val_acc_LR>acc:
        acc=std_mfcc_val_acc_LR
        best_C=C
LR_std_1024_128=LogisticRegression(penalty="l1", C=best_C, solver="liblinear",random_state=233)
LR_std_1024_128.fit(std_train_mfcc_data_1024_128,train_label)
std_test_pred_LR_1024_128=LR_1024_128.predict(std_test_mfcc_data_1024_128)
std_test_acc_LR_1024_128=accuracy_score(test_label,std_test_pred_LR_1024_128)
std_test_cr_LR_1024_128=classification_report(test_label,std_test_pred_LR_1024_128)
std_test_cm_LR_1024_128=confusion_matrix(test_label,std_test_pred_LR_1024_128)
std_LR_1024_128_coef=LR_std_1024_128.coef_
zero_per_LR_std_1024_128=np.sum(std_LR_1024_128_coef==0,axis=1)

## Multinomial Logistic Regression with L1 penalty with standard data
acc=0
for C in [0.01, 0.5, 0.1, 0.2, 1, 2, 5, 10,20]:
    MLR=LogisticRegression(penalty="l1", C=C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
    MLR.fit(std_train_mfcc_data_1024_128,train_label)
    std_mfcc_val_pred_MLR=MLR.predict(std_val_mfcc_data_1024_128)
    std_mfcc_val_acc_MLR=accuracy_score(val_label,std_mfcc_val_pred_MLR)
    if std_mfcc_val_acc_MLR>acc:
        acc=std_mfcc_val_acc_MLR
        best_C=C
MLR_std_1024_128=LogisticRegression(penalty="l1", C=best_C, solver="saga",multi_class="multinomial",max_iter=10000,random_state=233)
MLR_std_1024_128.fit(std_train_mfcc_data_1024_128,train_label)
std_test_pred_MLR_1024_128=MLR_std_1024_128.predict(std_test_mfcc_data_1024_128)
std_test_acc_MLR_1024_128=accuracy_score(test_label,std_test_pred_MLR_1024_128)
std_test_cr_MLR_1024_128=classification_report(test_label,std_test_pred_MLR_1024_128)
std_test_cm_MLR_1024_128=confusion_matrix(test_label,std_test_pred_MLR_1024_128)
std_MLR_1024_128_coef=MLR_std_1024_128.coef_
zero_per_MLR_std_1024_128=np.sum(std_MLR_1024_128_coef==0,axis=1)


file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Coefficients for Logistic Regression with L1 Penalty.png")
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1) 
sns.heatmap(np.abs(LR_2048_512_coef),cmap="gray_r")
plt.title("Data_2048_512")
plt.subplot(2,2,2) 
sns.heatmap(np.abs(LR_2048_256_coef),cmap="gray_r")
plt.title("Data_2048_256")
plt.subplot(2,2,3) 
sns.heatmap(np.abs(LR_1024_256_coef),cmap="gray_r")
plt.title("Data_1024_256")
plt.subplot(2,2,4) 
sns.heatmap(np.abs(LR_1024_128_coef),cmap="gray_r")
plt.title("Data_1024_128")
plt.suptitle("Coefficients for Logistic Regression with L1 Penalty", fontsize=16)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Coefficients for Logistic Multinomial Regression with L1 Penalty.png")
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1) 
sns.heatmap(np.abs(MLR_2048_512_coef),cmap="gray_r")
plt.title("Data_2048_512")
plt.subplot(2,2,2) 
sns.heatmap(np.abs(MLR_2048_256_coef),cmap="gray_r")
plt.title("Data_2048_256")
plt.subplot(2,2,3) 
sns.heatmap(np.abs(MLR_1024_256_coef),cmap="gray_r")
plt.title("Data_1024_256")
plt.subplot(2,2,4) 
sns.heatmap(np.abs(MLR_1024_128_coef),cmap="gray_r")
plt.title("Data_1024_128")
plt.suptitle("Coefficients for Logistic Multinomial Regression with L1 Penalty", fontsize=16)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Coefficients for Logistic Regression of Standard with L1 Penalty.png")
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1) 
sns.heatmap(np.abs(std_LR_2048_512_coef),cmap="gray_r")
plt.title("Data_2048_512")
plt.subplot(2,2,2) 
sns.heatmap(np.abs(std_LR_2048_256_coef),cmap="gray_r")
plt.title("Data_2048_256")
plt.subplot(2,2,3) 
sns.heatmap(np.abs(std_LR_1024_256_coef),cmap="gray_r")
plt.title("Data_1024_256")
plt.subplot(2,2,4) 
sns.heatmap(np.abs(std_LR_1024_128_coef),cmap="gray_r",)
plt.title("Data_1024_128")
plt.suptitle("Coefficients for Logistic Regression of Standard with L1 Penalty", fontsize=16)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Coefficients for Logistic Multinomial Regression of Standard with L1 Penalty.png")
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1) 
sns.heatmap(np.abs(std_MLR_2048_512_coef),cmap="gray_r")
plt.title("Data_2048_512")
plt.subplot(2,2,2) 
sns.heatmap(np.abs(std_MLR_2048_256_coef),cmap="gray_r")
plt.title("Data_2048_256")
plt.subplot(2,2,3) 
sns.heatmap(np.abs(std_MLR_1024_256_coef),cmap="gray_r")
plt.title("Data_1024_256")
plt.subplot(2,2,4) 
sns.heatmap(np.abs(std_MLR_1024_128_coef),cmap="gray_r")
plt.title("Data_1024_128")
plt.suptitle("Coefficients for Logistic Multinomial Regression of Standard with L1 Penalty", fontsize=16)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Coefficients for SVM with L1 penalty.png")
plt.figure(figsize=(20, 12))
plt.subplot(2,2,1) 
sns.heatmap(np.abs(svm_std_2048_512_coef),cmap="gray_r")
plt.title("Data_2048_512")
plt.subplot(2,2,2) 
sns.heatmap(np.abs(svm_std_2048_256_coef),cmap="gray_r")
plt.title("Data_2048_256")
plt.subplot(2,2,3) 
sns.heatmap(np.abs(svm_std_1024_256_coef),cmap="gray_r")
plt.title("Data_1024_256")
plt.subplot(2,2,4) 
sns.heatmap(np.abs(svm_std_1024_128_coef),cmap="gray_r")
plt.title("Data_1024_128")
plt.suptitle("Coefficients for SVM with L1 penalty", fontsize=16)
plt.savefig(os.path.join(file_path))

test_acc_knn_cosine_2048_512
test_acc_knn_euclidean_2048_512
test_acc_knn_L1_2048_512
test_acc_knn_std_euclidean_2048_512
test_acc_knn_std_L1_2048_512
test_acc_svml_std_2048_512
test_acc_LR_2048_512
test_acc_MLR_2048_512
std_test_acc_LR_2048_512
std_test_acc_MLR_2048_512

test_acc_knn_cosine_2048_256
test_acc_knn_euclidean_2048_256
test_acc_knn_L1_2048_256
test_acc_knn_std_euclidean_2048_256
test_acc_knn_std_L1_2048_256
test_acc_svml_std_2048_256
test_acc_LR_2048_256
test_acc_MLR_2048_256
std_test_acc_LR_2048_256
std_test_acc_MLR_2048_256

test_acc_knn_cosine_1024_256
test_acc_knn_euclidean_1024_256
test_acc_knn_L1_1024_256
test_acc_knn_std_euclidean_1024_256
test_acc_knn_std_L1_1024_256
test_acc_svml_std_1024_256
test_acc_LR_1024_256
test_acc_MLR_1024_256
std_test_acc_LR_1024_256
std_test_acc_MLR_1024_256

test_acc_knn_cosine_1024_128
test_acc_knn_euclidean_1024_128
test_acc_knn_L1_1024_128
test_acc_knn_std_euclidean_1024_128
test_acc_knn_std_L1_1024_128
test_acc_svml_std_1024_128
test_acc_LR_1024_128
test_acc_MLR_1024_128
std_test_acc_LR_1024_128
std_test_acc_MLR_1024_128



###########################Shapelet#########################

mfcc0_mt_2048_512=shapelet.MultiTree(MAXLEN=25,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mfcc0_mt_2048_512.fit(train_mfcc0_2048_512,train_label)
with open("mfcc0_mt_2048_512.pkl","wb") as f:
    pickle.dump(mfcc0_mt_2048_512,f)
with open("mfcc0_mt_2048_512.pkl","rb") as f:
    mfcc0_mt_2048_512=pickle.load(f)
train_pred_shapelet_2048_512=mfcc0_mt_2048_512.predict(train_mfcc0_2048_512) 
train_acc_shapelet_2048_512=accuracy_score(train_label,train_pred_shapelet_2048_512)
val_pred_shapelet_2048_512=mfcc0_mt_2048_512.predict(val_mfcc0_2048_512) 
val_acc_shapelet_2048_512=accuracy_score(val_label,val_pred_shapelet_2048_512)

test_pred_shapelet_2048_512=mfcc0_mt_2048_512.predict(test_mfcc0_2048_512) 
test_acc_shapelet_2048_512=accuracy_score(test_label,test_pred_shapelet_2048_512)
test_cm_shapelet_2048_512=confusion_matrix(test_label,test_pred_shapelet_2048_512)
test_cr_shapelet_2048_512=classification_report(test_label,test_pred_shapelet_2048_512)

mfcc0_mt_2048_256=shapelet.MultiTree(MAXLEN=25,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mfcc0_mt_2048_256.fit(train_mfcc0_2048_256,train_label)
with open("mfcc0_mt_2048_256.pkl","wb") as f:
    pickle.dump(mfcc0_mt_2048_256,f) 
with open("mfcc0_mt_2048_256.pkl","rb") as f:
    mfcc0_mt_2048_256=pickle.load(f) 
train_pred_shapelet_2048_256=mfcc0_mt_2048_256.predict(train_mfcc0_2048_256) 
train_acc_shapelet_2048_256=accuracy_score(train_label,train_pred_shapelet_2048_256)
val_pred_shapelet_2048_256=mfcc0_mt_2048_256.predict(val_mfcc0_2048_256) 
val_acc_shapelet_2048_256=accuracy_score(val_label,val_pred_shapelet_2048_256)
structure_2048_256=mfcc0_mt_2048_256.get_tree_structure()

test_pred_shapelet_2048_256=mfcc0_mt_2048_256.predict(test_mfcc0_2048_256) 
test_acc_shapelet_2048_256=accuracy_score(test_label,test_pred_shapelet_2048_256)
test_cm_shapelet_2048_256=confusion_matrix(test_label,test_pred_shapelet_2048_256)
test_cr_shapelet_2048_256=classification_report(test_label,test_pred_shapelet_2048_256)

mfcc0_mt_1024_256=shapelet.MultiTree(MAXLEN=25,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mfcc0_mt_1024_256.fit(train_mfcc0_1024_256,train_label)
with open("mfcc0_mt_1024_256.pkl","wb") as f:
    pickle.dump(mfcc0_mt_1024_256,f) 
with open("mfcc0_mt_1024_256.pkl","rb") as f:
    mfcc0_mt_1024_256=pickle.load(f) 
train_pred_shapelet_1024_256=mfcc0_mt_1024_256.predict(train_mfcc0_1024_256) 
train_acc_shapelet_1024_256=accuracy_score(train_label,train_pred_shapelet_1024_256)
val_pred_shapelet_1024_256=mfcc0_mt_1024_256.predict(val_mfcc0_1024_256) 
val_acc_shapelet_1024_256=accuracy_score(val_label,val_pred_shapelet_1024_256)

test_pred_shapelet_1024_256=mfcc0_mt_1024_256.predict(test_mfcc0_1024_256) 
test_acc_shapelet_1024_256=accuracy_score(test_label,test_pred_shapelet_1024_256)
test_cm_shapelet_1024_256=confusion_matrix(test_label,test_pred_shapelet_1024_256)
test_cr_shapelet_1024_256=classification_report(test_label,test_pred_shapelet_1024_256)

mfcc0_mt_1024_128=shapelet.MultiTree(MAXLEN=25,MINLEN=3,max_depth=3,min_samples_split=5)
with tf.device('/GPU:0'):
    mfcc0_mt_1024_128.fit(train_mfcc0_1024_128,train_label)
with open("mfcc0_mt_1024_128.pkl","wb") as f:
    pickle.dump(mfcc0_mt_1024_128,f) 
with open("mfcc0_mt_1024_128.pkl","rb") as f:
    mfcc0_mt_1024_128=pickle.load(f) 
train_pred_shapelet_1024_128=mfcc0_mt_1024_128.predict(train_mfcc0_1024_128) 
train_acc_shapelet_1024_128=accuracy_score(train_label,train_pred_shapelet_1024_128)
val_pred_shapelet_1024_128=mfcc0_mt_1024_128.predict(val_mfcc0_1024_128) 
val_acc_shapelet_1024_128=accuracy_score(val_label,val_pred_shapelet_1024_128)   

test_pred_shapelet_1024_128=mfcc0_mt_1024_128.predict(test_mfcc0_1024_128) 
test_acc_shapelet_1024_128=accuracy_score(test_label,test_pred_shapelet_1024_128)   
test_cm_shapelet_1024_128=confusion_matrix(test_label,test_pred_shapelet_1024_128)
test_cr_shapelet_1024_128=classification_report(test_label,test_pred_shapelet_1024_128)

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Confusion matrix of the Best Model.png") 
plt.figure(figsize=(6, 4))
sns.heatmap(std_test_cm_MLR_1024_128, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix of the Best Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(file_path))
plt.close()

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Confusion matrix of the Best Shapelet Model.png") 
plt.figure(figsize=(6, 4))
sns.heatmap(test_cm_shapelet_2048_256, annot=True, fmt='d', cmap='Reds')
plt.title('Confusion Matrix of the Best Shapelet Model')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig(os.path.join(file_path))
plt.close()

test_lab=[int(i) for i in test_label]
type_colors = {0: "lightblue", 1: "blue", 2: "green", 3: "violet"}
type_music=["Classical" ,"Disco", "Hiphop", "Mental"]
file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","Shapelets on Test Set.png") 
plt.figure(figsize=(10, 6))
for i in range(len(test_mfcc0_2048_256)):
    plt.plot(test_mfcc0_2048_256[i],color=type_colors[test_lab[i]], alpha=0.6,label=type_music[test_lab[i]])
plt.plot(structure_2048_256["feature"], color='red', alpha=0.6,label="Shaplet")
plt.plot(structure_2048_256["sub_nodes"][1]["feature"], color='red', alpha=0.6,label="Shaplet")
plt.plot(structure_2048_256["sub_nodes"][1]["sub_nodes"][0]["feature"], color='red', alpha=0.6,label="Shaplet")
plt.plot(structure_2048_256["sub_nodes"][1]["sub_nodes"][1]["feature"], color='red', alpha=0.6,label="Shaplet")
handles, labels = plt.gca().get_legend_handles_labels()
unique_labels = dict(zip(labels, handles))
plt.legend(unique_labels.values(), unique_labels.keys(), title="Legend",bbox_to_anchor=(1.05, 1),loc="upper left",borderaxespad=0.001)
plt.title("Shapelets on Test Set")
plt.tight_layout(rect=[0, 0, 0.975, 1])
plt.savefig(os.path.join(file_path))
plt.close()
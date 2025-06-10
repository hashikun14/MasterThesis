library(reticulate)
library(uniLasso)
use_condaenv("tf_gpu")
pickle<-import("pickle")
pd <- import("pandas")
all<-py_load_object("C:/scuola using/major/master thesis/trials/mfcc_features.pkl")
lab1 <- c(rep(1,100),rep(0,100),rep(0,100),rep(0,100))

audio_mfcc_feature_20s_2048_512<-all$audio_mfcc_feature_20s_2048_512
audio_mfcc_feature_20s_2048_256<-all$audio_mfcc_feature_20s_2048_256
audio_mfcc_feature_20s_1024_256<-all$audio_mfcc_feature_20s_1024_256
audio_mfcc_feature_20s_1024_128<-all$audio_mfcc_feature_20s_1024_128

unilassuo_2048_512<-uniLasso(audio_mfcc_feature_20s_2048_512,lab1,family="binomial",loo=T,
                             lambda=0.01)
plot(unilassuo_2048_512)
sum(coef(unilassuo_2048_512)==0)

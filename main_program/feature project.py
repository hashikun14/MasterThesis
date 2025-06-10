import os, librosa, math, pickle
import pandas as pd
import numpy as np
from ClassStatistics import RowStatistics, TimeseriesFeaturesTF

project_root=os.path.dirname(os.getcwd())

def get_subfolders(path):
    return [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]

def readfile(folder):
    data_list=[]
    for file in os.listdir(folder):
        if file.endswith(".wav"):
            file_path=os.path.join(folder, file)
            audio,sr=librosa.load(file_path,sr=None,mono=True,offset=5,duration=20)
            data_list.append({"Filename":file,"SampleRate":sr,"AudioData":audio})
    df=pd.DataFrame(data_list)
    return df

folders = get_subfolders(project_root+'\dataset\genres_original')
all_audio_df=pd.DataFrame()
for i in folders:
    trial=readfile(project_root+"\dataset\genres_original\ "[:-1] +i)
    all_audio_df=pd.concat([all_audio_df,trial],axis=0)    
all_audio_data=all_audio_df[["AudioData"]].set_index(all_audio_df["Filename"].to_numpy())
sample_rate=all_audio_df.iloc[0,1]

def mfcc_feature(n_fft,hop_length,window_length,sample_rate=22050):
    audio_mfcc_feature=pd.DataFrame()
    for i in range(0,400):
        mfcc0=librosa.feature.mfcc(y=all_audio_data.iloc[i,0],sr=sample_rate,n_mfcc=13,
                                hop_length=hop_length,n_fft=n_fft,n_mels=128)
        mfcc0=np.array(mfcc0, dtype=np.float64) 
        pass_df=pd.DataFrame()
        lungo=math.ceil(((sample_rate*window_length-n_fft)/hop_length)+1)
        for j in range(int(20/window_length)): # lungo comes from the formula n=((sample_per_window-n_fft)/hop_length)+1
            mfcc_texture=mfcc0[:,0+j*lungo:lungo+j*lungo]
            passaggio=RowStatistics(mfcc_texture).summary()
            column_names = [f"{key}_{k}_{j}" for key in passaggio.keys() for k in range(13)] #The ith MFCCs and the statistics of jth texture window
            flattened_data = np.concatenate(list(passaggio.values())).reshape(1, -1)
            pass_df_pass = pd.DataFrame(flattened_data, columns=column_names)
            pass_df=pd.concat([pass_df,pass_df_pass],axis=1)
            
        pass_df.index=[all_audio_data.index[i][:-4]]
        audio_mfcc_feature=pd.concat([audio_mfcc_feature,pass_df],axis=0)
    return audio_mfcc_feature

audio_mfcc_feature_20s_2048_512=mfcc_feature(n_fft=2048,hop_length=512,window_length=1)
audio_mfcc_feature_20s_2048_256=mfcc_feature(n_fft=2048,hop_length=256,window_length=1)
audio_mfcc_feature_20s_1024_256=mfcc_feature(n_fft=1024,hop_length=256,window_length=1)
audio_mfcc_feature_20s_1024_128=mfcc_feature(n_fft=1024,hop_length=128,window_length=1)

audio_mfcc_feature_50s_2048_512=mfcc_feature(n_fft=2048,hop_length=512,window_length=0.4)
audio_mfcc_feature_50s_2048_256=mfcc_feature(n_fft=2048,hop_length=256,window_length=0.4)
audio_mfcc_feature_50s_1024_256=mfcc_feature(n_fft=1024,hop_length=256,window_length=0.4)
audio_mfcc_feature_50s_1024_128=mfcc_feature(n_fft=1024,hop_length=128,window_length=0.4)

all_={
    "audio_mfcc_feature_20s_2048_512": audio_mfcc_feature_20s_2048_512,
    "audio_mfcc_feature_20s_2048_256": audio_mfcc_feature_20s_2048_256,
    "audio_mfcc_feature_20s_1024_256": audio_mfcc_feature_20s_1024_256,
    "audio_mfcc_feature_20s_1024_128": audio_mfcc_feature_20s_1024_128,
    "audio_mfcc_feature_50s_2048_512": audio_mfcc_feature_50s_2048_512,
    "audio_mfcc_feature_50s_2048_256": audio_mfcc_feature_50s_2048_256,
    "audio_mfcc_feature_50s_1024_256": audio_mfcc_feature_50s_1024_256,
    "audio_mfcc_feature_50s_1024_128": audio_mfcc_feature_50s_1024_128,
    }

with open("mfcc_features.pkl","wb") as f:
    pickle.dump(all_, f)

###########################################The Following Part is NOT Used in Thesis############################################################

def ts_f(second):
    audio_ts_feature=pd.DataFrame()
    for j in range(0,400):
        tr1=TimeseriesFeaturesTF(all_audio_df.iloc[j,2], second).summary()
        idf=str(second).replace(".","")
        coln=[f"{key}_{i}_{idf}" for key in tr1.keys() for i in range(int(20/second))]
        flatten=np.concatenate(list(tr1.values())).reshape(1, -1)
        tri_df = pd.DataFrame(flatten, columns=coln)
        tri_df.index=[all_audio_data.index[j][:-4]]
        
        acfname=[f"acf_{i}_{k}_{idf}" for i in range(int(20/second)) for k in range(10)]
        tt=pd.DataFrame(np.concatenate(TimeseriesFeaturesTF(all_audio_df.iloc[j,2],second).acf_second()))
        # the interpretation and treatment for the warning are demonstated as follows, please ignore it here
        #print(f"{j}")
        tt.fillna(1.0,inplace=True)
        tt.index=acfname
        ttt=pd.DataFrame(tt.T)
        ttt.index=[all_audio_data.index[j][:-4]]
        tri_df=pd.concat([tri_df,ttt],axis=1)        
        audio_ts_feature=pd.concat([audio_ts_feature,tri_df],axis=0)
    return audio_ts_feature
# Some of the music will stop for a while, that is the audio data of them are 0s for many of the corresponding frequencies and
# we cannot compute the autocorrelations of them because they are from a distribution that can be seen as a Dirac delta distribution
# which is degenerated and whose vairance is 0, resulting some errors when we use the function acf in the package of
# statsmodels.tsa.stattools which gives warning that the denominator computing acf is 0, which is the covariance of the serie. Hence I'd
# like to assign the na's to 1
ts_f_01=ts_f(second=0.1)
ts_f_04=ts_f(second=0.4)
ts_f_1=ts_f(second=1)
ts_f_3=ts_f(second=2)
ts_f_5=ts_f(second=5)

with open("ts_f_04.pkl","wb") as f:
    pickle.dump(ts_f_04,f) 

maxpool04=ts_f_04.filter(like="maxpool")
maxpool_04=[row.to_numpy() for _,row in maxpool04.iterrows()]
meanpool04=ts_f_04.filter(like="mean")
meanpool_04=[row.to_numpy() for _,row in meanpool04.iterrows()]

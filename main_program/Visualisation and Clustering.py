import pickle, os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster, cut_tree
from sklearn_extra.cluster import KMedoids
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
with open("mfcc_features.pkl","rb") as f:
    mfcc_features=pickle.load(f) 

project_root=os.path.dirname(os.getcwd())

lab=[]
for i in list(range(4)):
    lab+=[str(i)]*100
label=pd.DataFrame(lab)
audio_mfcc_feature_20s_2048_512=mfcc_features["audio_mfcc_feature_20s_2048_512"]
audio_mfcc_feature_20s_2048_256=mfcc_features["audio_mfcc_feature_20s_2048_256"]
audio_mfcc_feature_20s_1024_256=mfcc_features["audio_mfcc_feature_20s_1024_256"]
audio_mfcc_feature_20s_1024_128=mfcc_features["audio_mfcc_feature_20s_1024_128"]

audio_mfcc_feature_50s_2048_512=mfcc_features["audio_mfcc_feature_50s_2048_512"]
audio_mfcc_feature_50s_2048_256=mfcc_features["audio_mfcc_feature_50s_2048_256"]
audio_mfcc_feature_50s_1024_256=mfcc_features["audio_mfcc_feature_50s_1024_256"]
audio_mfcc_feature_50s_1024_128=mfcc_features["audio_mfcc_feature_50s_1024_128"]

eu_dist_mfcc_50s_2048_512=squareform(pdist(audio_mfcc_feature_50s_2048_512,metric="euclidean"))
L1_dist_mfcc_50s_2048_512=squareform(pdist(audio_mfcc_feature_50s_2048_512,metric="cityblock"))
cor_dist_mfcc_50s_2048_512=(0.5-audio_mfcc_feature_50s_2048_512.T.corr()/2).values

eu_dist_mfcc_50s_2048_256=squareform(pdist(audio_mfcc_feature_50s_2048_256,metric="euclidean"))
L1_dist_mfcc_50s_2048_256=squareform(pdist(audio_mfcc_feature_50s_2048_256,metric="cityblock"))
cor_dist_mfcc_50s_2048_256=(0.5-audio_mfcc_feature_50s_2048_256.T.corr()/2).values

eu_dist_mfcc_50s_1024_256=squareform(pdist(audio_mfcc_feature_50s_1024_256,metric="euclidean"))
L1_dist_mfcc_50s_1024_256=squareform(pdist(audio_mfcc_feature_50s_1024_256,metric="cityblock"))
cor_dist_mfcc_50s_1024_256=(0.5-audio_mfcc_feature_50s_1024_256.T.corr()/2).values

eu_dist_mfcc_50s_1024_128=squareform(pdist(audio_mfcc_feature_50s_1024_128,metric="euclidean"))
L1_dist_mfcc_50s_1024_128=squareform(pdist(audio_mfcc_feature_50s_1024_128,metric="cityblock"))
cor_dist_mfcc_50s_1024_128=(0.5-audio_mfcc_feature_50s_1024_128.T.corr()/2).values

eu_dist_mfcc_20s_2048_512=squareform(pdist(audio_mfcc_feature_20s_2048_512,metric="euclidean"))
L1_dist_mfcc_20s_2048_512=squareform(pdist(audio_mfcc_feature_20s_2048_512,metric="cityblock"))
cor_dist_mfcc_20s_2048_512=(0.5-audio_mfcc_feature_20s_2048_512.T.corr()/2).values

eu_dist_mfcc_20s_2048_256=squareform(pdist(audio_mfcc_feature_20s_2048_256,metric="euclidean"))
L1_dist_mfcc_20s_2048_256=squareform(pdist(audio_mfcc_feature_20s_2048_256,metric="cityblock"))
cor_dist_mfcc_20s_2048_256=(0.5-audio_mfcc_feature_20s_2048_256.T.corr()/2).values

eu_dist_mfcc_20s_1024_256=squareform(pdist(audio_mfcc_feature_20s_1024_256,metric="euclidean"))
L1_dist_mfcc_20s_1024_256=squareform(pdist(audio_mfcc_feature_20s_1024_256,metric="cityblock"))
cor_dist_mfcc_20s_1024_256=(0.5-audio_mfcc_feature_20s_1024_256.T.corr()/2).values

eu_dist_mfcc_20s_1024_128=squareform(pdist(audio_mfcc_feature_20s_1024_128,metric="euclidean"))
L1_dist_mfcc_20s_1024_128=squareform(pdist(audio_mfcc_feature_20s_1024_128,metric="cityblock"))
cor_dist_mfcc_20s_1024_128=(0.5-audio_mfcc_feature_20s_1024_128.T.corr()/2).values


########################################Visualisation###########################################
def mds_visualization(dist_matrix,main="",col="black"):
    mds=MDS(n_components=2,dissimilarity="precomputed",random_state=123,normalized_stress=False)
    mds_mfcc=mds.fit_transform(dist_matrix)
    print(round(mds.stress_/(dist_matrix**2).sum(),4))
    plt.scatter(mds_mfcc[:, 0], mds_mfcc[:, 1],s=10,c=col)
    plt.title(main)
    plt.xlabel("D1")
    plt.ylabel("D2")

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Euclidean Distance 50s.png")     
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_512,main="50s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_256,main="50s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_256,main="50s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_128,main="50s_1024_128 Features",col=label)
plt.suptitle("MDS of Euclidean Distance",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Mahanttan Distance 50s.png") 
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_512,main="50s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_256,main="50s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_256,main="50s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_128,main="50s_1024_128 Features",col=label)
plt.suptitle("MDS of Mahanttan Distance",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Correlation Dissimilarity 50s.png")
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_512,main="50s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_256,main="50s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_256,main="50s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_128,main="50s_1024_128 Features",col=label)
plt.suptitle("MDS of Correlation Dissimilarity",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Euclidean Distance 20s.png")
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_512,main="20s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_256,main="20s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_256,main="20s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_128,main="20s_1024_128 Features",col=label)
plt.suptitle("MDS of Euclidean Distance",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Manhattan Distance 20s.png")
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_512,main="20s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_256,main="20s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_256,main="20s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_128,main="20s_1024_128 Features",col=label)
plt.suptitle("MDS of Mahanttan Distance",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of Correlation Dissimilarity 20s.png")
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1) 
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_512,main="20s_2048_512 Features",col=label)
plt.subplot(2, 2, 2)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_256,main="20s_2048_256 Features",col=label)
plt.subplot(2, 2, 3) 
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_256,main="20s_1024_256 Features",col=label)
plt.subplot(2, 2, 4) 
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_128,main="20s_1024_128 Features",col=label)
plt.suptitle("MDS of Correlation Dissimilarity",fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(file_path))

########################################Clustering###########################################
def NK(cl, distmat):
    w=0
    K=1
    for i in range(2, 19): 
        c=cut_tree(cl, n_clusters=i).flatten()
        sil=silhouette_score(distmat, c, metric="precomputed")
        if w < sil:
            w=sil
            K=i
    return K
####Euclidean#####
conden_eu_50s_2048_512=squareform(eu_dist_mfcc_50s_2048_512)
H_com_eu_50s_2048_512=linkage(conden_eu_50s_2048_512, method='complete')
H_sin_eu_50s_2048_512=linkage(conden_eu_50s_2048_512, method='single')
H_ave_eu_50s_2048_512=linkage(conden_eu_50s_2048_512, method='average')

H_com_eu_50s_2048_512_K=NK(H_com_eu_50s_2048_512, eu_dist_mfcc_50s_2048_512)
H_sin_eu_50s_2048_512_K=NK(H_sin_eu_50s_2048_512, eu_dist_mfcc_50s_2048_512)
H_ave_eu_50s_2048_512_K=NK(H_ave_eu_50s_2048_512, eu_dist_mfcc_50s_2048_512)

H_com_eu_50s_2048_512_cl=fcluster(H_com_eu_50s_2048_512,t=H_com_eu_50s_2048_512_K, criterion="maxclust")
H_sin_eu_50s_2048_512_cl=fcluster(H_sin_eu_50s_2048_512,t=H_sin_eu_50s_2048_512_K, criterion="maxclust")
H_ave_eu_50s_2048_512_cl=fcluster(H_ave_eu_50s_2048_512,t=H_ave_eu_50s_2048_512_K, criterion="maxclust")

conden_eu_50s_2048_256=squareform(eu_dist_mfcc_50s_2048_256)
H_com_eu_50s_2048_256=linkage(conden_eu_50s_2048_256, method='complete')
H_sin_eu_50s_2048_256=linkage(conden_eu_50s_2048_256, method='single')
H_ave_eu_50s_2048_256=linkage(conden_eu_50s_2048_256, method='average')

H_com_eu_50s_2048_256_K=NK(H_com_eu_50s_2048_256, eu_dist_mfcc_50s_2048_256)
H_sin_eu_50s_2048_256_K=NK(H_sin_eu_50s_2048_256, eu_dist_mfcc_50s_2048_256)
H_ave_eu_50s_2048_256_K=NK(H_ave_eu_50s_2048_256, eu_dist_mfcc_50s_2048_256)

H_com_eu_50s_2048_256_cl=fcluster(H_com_eu_50s_2048_256,t=H_com_eu_50s_2048_256_K, criterion="maxclust")
H_sin_eu_50s_2048_256_cl=fcluster(H_sin_eu_50s_2048_256,t=H_sin_eu_50s_2048_256_K, criterion="maxclust")
H_ave_eu_50s_2048_256_cl=fcluster(H_ave_eu_50s_2048_256,t=H_ave_eu_50s_2048_256_K, criterion="maxclust")

conden_eu_50s_1024_256=squareform(eu_dist_mfcc_50s_1024_256)
H_com_eu_50s_1024_256=linkage(conden_eu_50s_1024_256, method='complete')
H_sin_eu_50s_1024_256=linkage(conden_eu_50s_1024_256, method='single')
H_ave_eu_50s_1024_256=linkage(conden_eu_50s_1024_256, method='average')

H_com_eu_50s_1024_256_K=NK(H_com_eu_50s_1024_256, eu_dist_mfcc_50s_1024_256)
H_sin_eu_50s_1024_256_K=NK(H_sin_eu_50s_1024_256, eu_dist_mfcc_50s_1024_256)
H_ave_eu_50s_1024_256_K=NK(H_ave_eu_50s_1024_256, eu_dist_mfcc_50s_1024_256)

H_com_eu_50s_1024_256_cl=fcluster(H_com_eu_50s_1024_256,t=H_com_eu_50s_1024_256_K, criterion="maxclust")
H_sin_eu_50s_1024_256_cl=fcluster(H_sin_eu_50s_1024_256,t=H_sin_eu_50s_1024_256_K, criterion="maxclust")
H_ave_eu_50s_1024_256_cl=fcluster(H_ave_eu_50s_1024_256,t=H_ave_eu_50s_1024_256_K, criterion="maxclust")

conden_eu_50s_1024_128=squareform(eu_dist_mfcc_50s_1024_128)
H_com_eu_50s_1024_128=linkage(conden_eu_50s_1024_128, method='complete')
H_sin_eu_50s_1024_128=linkage(conden_eu_50s_1024_128, method='single')
H_ave_eu_50s_1024_128=linkage(conden_eu_50s_1024_128, method='average')

H_com_eu_50s_1024_128_K=NK(H_com_eu_50s_1024_128, eu_dist_mfcc_50s_1024_128)
H_sin_eu_50s_1024_128_K=NK(H_sin_eu_50s_1024_128, eu_dist_mfcc_50s_1024_128)
H_ave_eu_50s_1024_128_K=NK(H_ave_eu_50s_1024_128, eu_dist_mfcc_50s_1024_128)

H_com_eu_50s_1024_128_cl=fcluster(H_com_eu_50s_1024_128,t=H_com_eu_50s_1024_128_K, criterion="maxclust")
H_sin_eu_50s_1024_128_cl=fcluster(H_sin_eu_50s_1024_128,t=H_sin_eu_50s_1024_128_K, criterion="maxclust")
H_ave_eu_50s_1024_128_cl=fcluster(H_ave_eu_50s_1024_128,t=H_ave_eu_50s_1024_128_K, criterion="maxclust")

conden_eu_20s_2048_512=squareform(eu_dist_mfcc_50s_2048_512)
H_com_eu_20s_2048_512=linkage(conden_eu_20s_2048_512, method='complete')
H_sin_eu_20s_2048_512=linkage(conden_eu_20s_2048_512, method='single')
H_ave_eu_20s_2048_512=linkage(conden_eu_20s_2048_512, method='average')

H_com_eu_20s_2048_512_K=NK(H_com_eu_20s_2048_512, eu_dist_mfcc_20s_2048_512)
H_sin_eu_20s_2048_512_K=NK(H_sin_eu_20s_2048_512, eu_dist_mfcc_20s_2048_512)
H_ave_eu_20s_2048_512_K=NK(H_ave_eu_20s_2048_512, eu_dist_mfcc_20s_2048_512)

H_com_eu_20s_2048_512_cl=fcluster(H_com_eu_20s_2048_512,t=H_com_eu_20s_2048_512_K, criterion="maxclust")
H_sin_eu_20s_2048_512_cl=fcluster(H_sin_eu_20s_2048_512,t=H_sin_eu_20s_2048_512_K, criterion="maxclust")
H_ave_eu_20s_2048_512_cl=fcluster(H_ave_eu_20s_2048_512,t=H_ave_eu_20s_2048_512_K, criterion="maxclust")

conden_eu_20s_2048_256=squareform(eu_dist_mfcc_50s_2048_256)
H_com_eu_20s_2048_256=linkage(conden_eu_20s_2048_256, method='complete')
H_sin_eu_20s_2048_256=linkage(conden_eu_20s_2048_256, method='single')
H_ave_eu_20s_2048_256=linkage(conden_eu_20s_2048_256, method='average')

H_com_eu_20s_2048_256_K=NK(H_com_eu_20s_2048_256, eu_dist_mfcc_20s_2048_256)
H_sin_eu_20s_2048_256_K=NK(H_sin_eu_20s_2048_256, eu_dist_mfcc_20s_2048_256)
H_ave_eu_20s_2048_256_K=NK(H_ave_eu_20s_2048_256, eu_dist_mfcc_20s_2048_256) ## 3

H_com_eu_20s_2048_256_cl=fcluster(H_com_eu_20s_2048_256,t=H_com_eu_20s_2048_256_K, criterion="maxclust")
H_sin_eu_20s_2048_256_cl=fcluster(H_sin_eu_20s_2048_256,t=H_sin_eu_20s_2048_256_K, criterion="maxclust")
H_ave_eu_20s_2048_256_cl=fcluster(H_ave_eu_20s_2048_256,t=H_ave_eu_20s_2048_256_K, criterion="maxclust")

conden_eu_20s_1024_256=squareform(eu_dist_mfcc_50s_1024_256)
H_com_eu_20s_1024_256=linkage(conden_eu_20s_1024_256, method='complete')
H_sin_eu_20s_1024_256=linkage(conden_eu_20s_1024_256, method='single')
H_ave_eu_20s_1024_256=linkage(conden_eu_20s_1024_256, method='average')

H_com_eu_20s_1024_256_K=NK(H_com_eu_20s_1024_256, eu_dist_mfcc_20s_1024_256)
H_sin_eu_20s_1024_256_K=NK(H_sin_eu_20s_1024_256, eu_dist_mfcc_20s_1024_256)
H_ave_eu_20s_1024_256_K=NK(H_ave_eu_20s_1024_256, eu_dist_mfcc_20s_1024_256)

H_com_eu_20s_1024_256_cl=fcluster(H_com_eu_20s_1024_256,t=H_com_eu_20s_1024_256_K, criterion="maxclust")
H_sin_eu_20s_1024_256_cl=fcluster(H_sin_eu_20s_1024_256,t=H_sin_eu_20s_1024_256_K, criterion="maxclust")
H_ave_eu_20s_1024_256_cl=fcluster(H_ave_eu_20s_1024_256,t=H_ave_eu_20s_1024_256_K, criterion="maxclust")

conden_eu_20s_1024_128=squareform(eu_dist_mfcc_50s_1024_128)
H_com_eu_20s_1024_128=linkage(conden_eu_20s_1024_128, method='complete')
H_sin_eu_20s_1024_128=linkage(conden_eu_20s_1024_128, method='single')
H_ave_eu_20s_1024_128=linkage(conden_eu_20s_1024_128, method='average')

H_com_eu_20s_1024_128_K=NK(H_com_eu_20s_1024_128, eu_dist_mfcc_20s_1024_128)
H_sin_eu_20s_1024_128_K=NK(H_sin_eu_20s_1024_128, eu_dist_mfcc_20s_1024_128) #3
H_ave_eu_20s_1024_128_K=NK(H_ave_eu_20s_1024_128, eu_dist_mfcc_20s_1024_128)

H_com_eu_20s_1024_128_cl=fcluster(H_com_eu_20s_1024_128,t=H_com_eu_20s_1024_128_K, criterion="maxclust")
H_sin_eu_20s_1024_128_cl=fcluster(H_sin_eu_20s_1024_128,t=H_sin_eu_20s_1024_128_K, criterion="maxclust")
H_ave_eu_20s_1024_128_cl=fcluster(H_ave_eu_20s_1024_128,t=H_ave_eu_20s_1024_128_K, criterion="maxclust")

#####L1#####

conden_L1_50s_2048_512=squareform(L1_dist_mfcc_50s_2048_512)
H_com_L1_50s_2048_512=linkage(conden_L1_50s_2048_512, method='complete')
H_sin_L1_50s_2048_512=linkage(conden_L1_50s_2048_512, method='single')
H_ave_L1_50s_2048_512=linkage(conden_L1_50s_2048_512, method='average')

H_com_L1_50s_2048_512_K=NK(H_com_L1_50s_2048_512, L1_dist_mfcc_50s_2048_512)
H_sin_L1_50s_2048_512_K=NK(H_sin_L1_50s_2048_512, L1_dist_mfcc_50s_2048_512)
H_ave_L1_50s_2048_512_K=NK(H_ave_L1_50s_2048_512, L1_dist_mfcc_50s_2048_512)

H_com_L1_50s_2048_512_cl=fcluster(H_com_L1_50s_2048_512,t=H_com_L1_50s_2048_512_K, criterion="maxclust")
H_sin_L1_50s_2048_512_cl=fcluster(H_sin_L1_50s_2048_512,t=H_sin_L1_50s_2048_512_K, criterion="maxclust")
H_ave_L1_50s_2048_512_cl=fcluster(H_ave_L1_50s_2048_512,t=H_ave_L1_50s_2048_512_K, criterion="maxclust")

conden_L1_50s_2048_256=squareform(L1_dist_mfcc_50s_2048_256)
H_com_L1_50s_2048_256=linkage(conden_L1_50s_2048_256, method='complete')
H_sin_L1_50s_2048_256=linkage(conden_L1_50s_2048_256, method='single')
H_ave_L1_50s_2048_256=linkage(conden_L1_50s_2048_256, method='average')

H_com_L1_50s_2048_256_K=NK(H_com_L1_50s_2048_256, L1_dist_mfcc_50s_2048_256)
H_sin_L1_50s_2048_256_K=NK(H_sin_L1_50s_2048_256, L1_dist_mfcc_50s_2048_256)
H_ave_L1_50s_2048_256_K=NK(H_ave_L1_50s_2048_256, L1_dist_mfcc_50s_2048_256)

H_com_L1_50s_2048_256_cl=fcluster(H_com_L1_50s_2048_256,t=H_com_L1_50s_2048_256_K, criterion="maxclust")
H_sin_L1_50s_2048_256_cl=fcluster(H_sin_L1_50s_2048_256,t=H_sin_L1_50s_2048_256_K, criterion="maxclust")
H_ave_L1_50s_2048_256_cl=fcluster(H_ave_L1_50s_2048_256,t=H_ave_L1_50s_2048_256_K, criterion="maxclust")

conden_L1_50s_1024_256=squareform(L1_dist_mfcc_50s_1024_256)
H_com_L1_50s_1024_256=linkage(conden_L1_50s_1024_256, method='complete')
H_sin_L1_50s_1024_256=linkage(conden_L1_50s_1024_256, method='single')
H_ave_L1_50s_1024_256=linkage(conden_L1_50s_1024_256, method='average')

H_com_L1_50s_1024_256_K=NK(H_com_L1_50s_1024_256, L1_dist_mfcc_50s_1024_256)
H_sin_L1_50s_1024_256_K=NK(H_sin_L1_50s_1024_256, L1_dist_mfcc_50s_1024_256)
H_ave_L1_50s_1024_256_K=NK(H_ave_L1_50s_1024_256, L1_dist_mfcc_50s_1024_256)

H_com_L1_50s_1024_256_cl=fcluster(H_com_L1_50s_1024_256,t=H_com_L1_50s_1024_256_K, criterion="maxclust")
H_sin_L1_50s_1024_256_cl=fcluster(H_sin_L1_50s_1024_256,t=H_sin_L1_50s_1024_256_K, criterion="maxclust")
H_ave_L1_50s_1024_256_cl=fcluster(H_ave_L1_50s_1024_256,t=H_ave_L1_50s_1024_256_K, criterion="maxclust")

conden_L1_50s_1024_128=squareform(L1_dist_mfcc_50s_1024_128)
H_com_L1_50s_1024_128=linkage(conden_L1_50s_1024_128, method='complete')
H_sin_L1_50s_1024_128=linkage(conden_L1_50s_1024_128, method='single')
H_ave_L1_50s_1024_128=linkage(conden_L1_50s_1024_128, method='average')

H_com_L1_50s_1024_128_K=NK(H_com_L1_50s_1024_128, L1_dist_mfcc_50s_1024_128)
H_sin_L1_50s_1024_128_K=NK(H_sin_L1_50s_1024_128, L1_dist_mfcc_50s_1024_128)
H_ave_L1_50s_1024_128_K=NK(H_ave_L1_50s_1024_128, L1_dist_mfcc_50s_1024_128)

H_com_L1_50s_1024_128_cl=fcluster(H_com_L1_50s_1024_128,t=H_com_L1_50s_1024_128_K, criterion="maxclust")
H_sin_L1_50s_1024_128_cl=fcluster(H_sin_L1_50s_1024_128,t=H_sin_L1_50s_1024_128_K, criterion="maxclust")
H_ave_L1_50s_1024_128_cl=fcluster(H_ave_L1_50s_1024_128,t=H_ave_L1_50s_1024_128_K, criterion="maxclust")

conden_L1_20s_2048_512=squareform(L1_dist_mfcc_50s_2048_512)
H_com_L1_20s_2048_512=linkage(conden_L1_20s_2048_512, method='complete')
H_sin_L1_20s_2048_512=linkage(conden_L1_20s_2048_512, method='single')
H_ave_L1_20s_2048_512=linkage(conden_L1_20s_2048_512, method='average')

H_com_L1_20s_2048_512_K=NK(H_com_L1_20s_2048_512, L1_dist_mfcc_20s_2048_512)
H_sin_L1_20s_2048_512_K=NK(H_sin_L1_20s_2048_512, L1_dist_mfcc_20s_2048_512)
H_ave_L1_20s_2048_512_K=NK(H_ave_L1_20s_2048_512, L1_dist_mfcc_20s_2048_512)

H_com_L1_20s_2048_512_cl=fcluster(H_com_L1_20s_2048_512,t=H_com_L1_20s_2048_512_K, criterion="maxclust")
H_sin_L1_20s_2048_512_cl=fcluster(H_sin_L1_20s_2048_512,t=H_sin_L1_20s_2048_512_K, criterion="maxclust")
H_ave_L1_20s_2048_512_cl=fcluster(H_ave_L1_20s_2048_512,t=H_ave_L1_20s_2048_512_K, criterion="maxclust")

conden_L1_20s_2048_256=squareform(L1_dist_mfcc_50s_2048_256)
H_com_L1_20s_2048_256=linkage(conden_L1_20s_2048_256, method='complete')
H_sin_L1_20s_2048_256=linkage(conden_L1_20s_2048_256, method='single')
H_ave_L1_20s_2048_256=linkage(conden_L1_20s_2048_256, method='average')

H_com_L1_20s_2048_256_K=NK(H_com_L1_20s_2048_256, L1_dist_mfcc_20s_2048_256)
H_sin_L1_20s_2048_256_K=NK(H_sin_L1_20s_2048_256, L1_dist_mfcc_20s_2048_256)
H_ave_L1_20s_2048_256_K=NK(H_ave_L1_20s_2048_256, L1_dist_mfcc_20s_2048_256)

H_com_L1_20s_2048_256_cl=fcluster(H_com_L1_20s_2048_256,t=H_com_L1_20s_2048_256_K, criterion="maxclust")
H_sin_L1_20s_2048_256_cl=fcluster(H_sin_L1_20s_2048_256,t=H_sin_L1_20s_2048_256_K, criterion="maxclust")
H_ave_L1_20s_2048_256_cl=fcluster(H_ave_L1_20s_2048_256,t=H_ave_L1_20s_2048_256_K, criterion="maxclust")

conden_L1_20s_1024_256=squareform(L1_dist_mfcc_50s_1024_256)
H_com_L1_20s_1024_256=linkage(conden_L1_20s_1024_256, method='complete')
H_sin_L1_20s_1024_256=linkage(conden_L1_20s_1024_256, method='single')
H_ave_L1_20s_1024_256=linkage(conden_L1_20s_1024_256, method='average')

H_com_L1_20s_1024_256_K=NK(H_com_L1_20s_1024_256, L1_dist_mfcc_20s_1024_256)
H_sin_L1_20s_1024_256_K=NK(H_sin_L1_20s_1024_256, L1_dist_mfcc_20s_1024_256)
H_ave_L1_20s_1024_256_K=NK(H_ave_L1_20s_1024_256, L1_dist_mfcc_20s_1024_256)

H_com_L1_20s_1024_256_cl=fcluster(H_com_L1_20s_1024_256,t=H_com_L1_20s_1024_256_K, criterion="maxclust")
H_sin_L1_20s_1024_256_cl=fcluster(H_sin_L1_20s_1024_256,t=H_sin_L1_20s_1024_256_K, criterion="maxclust")
H_ave_L1_20s_1024_256_cl=fcluster(H_ave_L1_20s_1024_256,t=H_ave_L1_20s_1024_256_K, criterion="maxclust")

conden_L1_20s_1024_128=squareform(L1_dist_mfcc_50s_1024_128)
H_com_L1_20s_1024_128=linkage(conden_L1_20s_1024_128, method='complete')
H_sin_L1_20s_1024_128=linkage(conden_L1_20s_1024_128, method='single')
H_ave_L1_20s_1024_128=linkage(conden_L1_20s_1024_128, method='average')

H_com_L1_20s_1024_128_K=NK(H_com_L1_20s_1024_128, L1_dist_mfcc_20s_1024_128)
H_sin_L1_20s_1024_128_K=NK(H_sin_L1_20s_1024_128, L1_dist_mfcc_20s_1024_128)
H_ave_L1_20s_1024_128_K=NK(H_ave_L1_20s_1024_128, L1_dist_mfcc_20s_1024_128)

H_com_L1_20s_1024_128_cl=fcluster(H_com_L1_20s_1024_128,t=H_com_L1_20s_1024_128_K, criterion="maxclust")
H_sin_L1_20s_1024_128_cl=fcluster(H_sin_L1_20s_1024_128,t=H_sin_L1_20s_1024_128_K, criterion="maxclust")
H_ave_L1_20s_1024_128_cl=fcluster(H_ave_L1_20s_1024_128,t=H_ave_L1_20s_1024_128_K, criterion="maxclust")

#####Cor#####

conden_cor_50s_2048_512=squareform(cor_dist_mfcc_50s_2048_512)
H_com_cor_50s_2048_512=linkage(conden_cor_50s_2048_512, method='complete')
H_sin_cor_50s_2048_512=linkage(conden_cor_50s_2048_512, method='single')
H_ave_cor_50s_2048_512=linkage(conden_cor_50s_2048_512, method='average')

H_com_cor_50s_2048_512_K=NK(H_com_cor_50s_2048_512, cor_dist_mfcc_50s_2048_512) ##3
H_sin_cor_50s_2048_512_K=NK(H_sin_cor_50s_2048_512, cor_dist_mfcc_50s_2048_512)
H_ave_cor_50s_2048_512_K=NK(H_ave_cor_50s_2048_512, cor_dist_mfcc_50s_2048_512)

H_com_cor_50s_2048_512_cl=fcluster(H_com_cor_50s_2048_512,t=H_com_cor_50s_2048_512_K, criterion="maxclust")
H_sin_cor_50s_2048_512_cl=fcluster(H_sin_cor_50s_2048_512,t=H_sin_cor_50s_2048_512_K, criterion="maxclust")
H_ave_cor_50s_2048_512_cl=fcluster(H_ave_cor_50s_2048_512,t=H_ave_cor_50s_2048_512_K, criterion="maxclust")

conden_cor_50s_2048_256=squareform(cor_dist_mfcc_50s_2048_256)
H_com_cor_50s_2048_256=linkage(conden_cor_50s_2048_256, method='complete')
H_sin_cor_50s_2048_256=linkage(conden_cor_50s_2048_256, method='single')
H_ave_cor_50s_2048_256=linkage(conden_cor_50s_2048_256, method='average')

H_com_cor_50s_2048_256_K=NK(H_com_cor_50s_2048_256, cor_dist_mfcc_50s_2048_256) ##4
H_sin_cor_50s_2048_256_K=NK(H_sin_cor_50s_2048_256, cor_dist_mfcc_50s_2048_256)
H_ave_cor_50s_2048_256_K=NK(H_ave_cor_50s_2048_256, cor_dist_mfcc_50s_2048_256)

H_com_cor_50s_2048_256_cl=fcluster(H_com_cor_50s_2048_256,t=H_com_cor_50s_2048_256_K, criterion="maxclust")
H_sin_cor_50s_2048_256_cl=fcluster(H_sin_cor_50s_2048_256,t=H_sin_cor_50s_2048_256_K, criterion="maxclust")
H_ave_cor_50s_2048_256_cl=fcluster(H_ave_cor_50s_2048_256,t=H_ave_cor_50s_2048_256_K, criterion="maxclust")

conden_cor_50s_1024_256=squareform(cor_dist_mfcc_50s_1024_256)
H_com_cor_50s_1024_256=linkage(conden_cor_50s_1024_256, method='complete')
H_sin_cor_50s_1024_256=linkage(conden_cor_50s_1024_256, method='single')
H_ave_cor_50s_1024_256=linkage(conden_cor_50s_1024_256, method='average')

H_com_cor_50s_1024_256_K=NK(H_com_cor_50s_1024_256, cor_dist_mfcc_50s_1024_256) ## 3
H_sin_cor_50s_1024_256_K=NK(H_sin_cor_50s_1024_256, cor_dist_mfcc_50s_1024_256)
H_ave_cor_50s_1024_256_K=NK(H_ave_cor_50s_1024_256, cor_dist_mfcc_50s_1024_256)

H_com_cor_50s_1024_256_cl=fcluster(H_com_cor_50s_1024_256,t=H_com_cor_50s_1024_256_K, criterion="maxclust")
H_sin_cor_50s_1024_256_cl=fcluster(H_sin_cor_50s_1024_256,t=H_sin_cor_50s_1024_256_K, criterion="maxclust")
H_ave_cor_50s_1024_256_cl=fcluster(H_ave_cor_50s_1024_256,t=H_ave_cor_50s_1024_256_K, criterion="maxclust")

conden_cor_50s_1024_128=squareform(cor_dist_mfcc_50s_1024_128)
H_com_cor_50s_1024_128=linkage(conden_cor_50s_1024_128, method='complete')
H_sin_cor_50s_1024_128=linkage(conden_cor_50s_1024_128, method='single')
H_ave_cor_50s_1024_128=linkage(conden_cor_50s_1024_128, method='average')

H_com_cor_50s_1024_128_K=NK(H_com_cor_50s_1024_128, cor_dist_mfcc_50s_1024_128)
H_sin_cor_50s_1024_128_K=NK(H_sin_cor_50s_1024_128, cor_dist_mfcc_50s_1024_128)
H_ave_cor_50s_1024_128_K=NK(H_ave_cor_50s_1024_128, cor_dist_mfcc_50s_1024_128)

H_com_cor_50s_1024_128_cl=fcluster(H_com_cor_50s_1024_128,t=H_com_cor_50s_1024_128_K, criterion="maxclust")
H_sin_cor_50s_1024_128_cl=fcluster(H_sin_cor_50s_1024_128,t=H_sin_cor_50s_1024_128_K, criterion="maxclust")
H_ave_cor_50s_1024_128_cl=fcluster(H_ave_cor_50s_1024_128,t=H_ave_cor_50s_1024_128_K, criterion="maxclust")

conden_cor_20s_2048_512=squareform(cor_dist_mfcc_50s_2048_512)
H_com_cor_20s_2048_512=linkage(conden_cor_20s_2048_512, method='complete')
H_sin_cor_20s_2048_512=linkage(conden_cor_20s_2048_512, method='single')
H_ave_cor_20s_2048_512=linkage(conden_cor_20s_2048_512, method='average')

H_com_cor_20s_2048_512_K=NK(H_com_cor_20s_2048_512, cor_dist_mfcc_20s_2048_512) ##3
H_sin_cor_20s_2048_512_K=NK(H_sin_cor_20s_2048_512, cor_dist_mfcc_20s_2048_512)
H_ave_cor_20s_2048_512_K=NK(H_ave_cor_20s_2048_512, cor_dist_mfcc_20s_2048_512)

H_com_cor_20s_2048_512_cl=fcluster(H_com_cor_20s_2048_512,t=H_com_cor_20s_2048_512_K, criterion="maxclust")
H_sin_cor_20s_2048_512_cl=fcluster(H_sin_cor_20s_2048_512,t=H_sin_cor_20s_2048_512_K, criterion="maxclust")
H_ave_cor_20s_2048_512_cl=fcluster(H_ave_cor_20s_2048_512,t=H_ave_cor_20s_2048_512_K, criterion="maxclust")

conden_cor_20s_2048_256=squareform(cor_dist_mfcc_50s_2048_256)
H_com_cor_20s_2048_256=linkage(conden_cor_20s_2048_256, method='complete')
H_sin_cor_20s_2048_256=linkage(conden_cor_20s_2048_256, method='single')
H_ave_cor_20s_2048_256=linkage(conden_cor_20s_2048_256, method='average')

H_com_cor_20s_2048_256_K=NK(H_com_cor_20s_2048_256, cor_dist_mfcc_20s_2048_256)
H_sin_cor_20s_2048_256_K=NK(H_sin_cor_20s_2048_256, cor_dist_mfcc_20s_2048_256)
H_ave_cor_20s_2048_256_K=NK(H_ave_cor_20s_2048_256, cor_dist_mfcc_20s_2048_256) 

H_com_cor_20s_2048_256_cl=fcluster(H_com_cor_20s_2048_256,t=H_com_cor_20s_2048_256_K, criterion="maxclust")
H_sin_cor_20s_2048_256_cl=fcluster(H_sin_cor_20s_2048_256,t=H_sin_cor_20s_2048_256_K, criterion="maxclust")
H_ave_cor_20s_2048_256_cl=fcluster(H_ave_cor_20s_2048_256,t=H_ave_cor_20s_2048_256_K, criterion="maxclust")

conden_cor_20s_1024_256=squareform(cor_dist_mfcc_50s_1024_256)
H_com_cor_20s_1024_256=linkage(conden_cor_20s_1024_256, method='complete')
H_sin_cor_20s_1024_256=linkage(conden_cor_20s_1024_256, method='single')
H_ave_cor_20s_1024_256=linkage(conden_cor_20s_1024_256, method='average')

H_com_cor_20s_1024_256_K=NK(H_com_cor_20s_1024_256, cor_dist_mfcc_20s_1024_256) ##3
H_sin_cor_20s_1024_256_K=NK(H_sin_cor_20s_1024_256, cor_dist_mfcc_20s_1024_256)
H_ave_cor_20s_1024_256_K=NK(H_ave_cor_20s_1024_256, cor_dist_mfcc_20s_1024_256)

H_com_cor_20s_1024_256_cl=fcluster(H_com_cor_20s_1024_256,t=H_com_cor_20s_1024_256_K, criterion="maxclust")
H_sin_cor_20s_1024_256_cl=fcluster(H_sin_cor_20s_1024_256,t=H_sin_cor_20s_1024_256_K, criterion="maxclust")
H_ave_cor_20s_1024_256_cl=fcluster(H_ave_cor_20s_1024_256,t=H_ave_cor_20s_1024_256_K, criterion="maxclust")

conden_cor_20s_1024_128=squareform(cor_dist_mfcc_50s_1024_128)
H_com_cor_20s_1024_128=linkage(conden_cor_20s_1024_128, method='complete')
H_sin_cor_20s_1024_128=linkage(conden_cor_20s_1024_128, method='single')
H_ave_cor_20s_1024_128=linkage(conden_cor_20s_1024_128, method='average')

H_com_cor_20s_1024_128_K=NK(H_com_cor_20s_1024_128, cor_dist_mfcc_20s_1024_128)
H_sin_cor_20s_1024_128_K=NK(H_sin_cor_20s_1024_128, cor_dist_mfcc_20s_1024_128)
H_ave_cor_20s_1024_128_K=NK(H_ave_cor_20s_1024_128, cor_dist_mfcc_20s_1024_128)

H_com_cor_20s_1024_128_cl=fcluster(H_com_cor_20s_1024_128,t=H_com_cor_20s_1024_128_K, criterion="maxclust")
H_sin_cor_20s_1024_128_cl=fcluster(H_sin_cor_20s_1024_128,t=H_sin_cor_20s_1024_128_K, criterion="maxclust")
H_ave_cor_20s_1024_128_cl=fcluster(H_ave_cor_20s_1024_128,t=H_ave_cor_20s_1024_128_K, criterion="maxclust")

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_512 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_512,main="Euclidean Distance",col=H_com_eu_50s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_512,main="Manhattan Distance",col=H_com_L1_50s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_512,main="Correlation Dissimilarity",col=H_com_cor_50s_2048_512_cl)
plt.suptitle("MDS of 50s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_256 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_256,main="Euclidean Distance",col=H_com_eu_50s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_256,main="Manhattan Distance",col=H_com_L1_50s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_256,main="Correlation Dissimilarity",col=H_com_cor_50s_2048_256_cl)
plt.suptitle("MDS of 50s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_256 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_256,main="Euclidean Distance",col=H_com_eu_50s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_256,main="Manhattan Distance",col=H_com_L1_50s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_256,main="Correlation Dissimilarity",col=H_com_cor_50s_1024_256_cl)
plt.suptitle("MDS of 50s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_128 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_128,main="Euclidean Distance",col=H_com_eu_50s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_128,main="Manhattan Distance",col=H_com_L1_50s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_128,main="Correlation Dissimilarity",col=H_com_cor_50s_1024_128_cl)
plt.suptitle("MDS of 50s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_512 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_512,main="Euclidean Distance",col=H_sin_eu_50s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_512,main="Manhattan Distance",col=H_sin_L1_50s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_512,main="Correlation Dissimilarity",col=H_sin_cor_50s_2048_512_cl)
plt.suptitle("MDS of 50s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_256 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_256,main="Euclidean Distance",col=H_sin_eu_50s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_256,main="Manhattan Distance",col=H_sin_L1_50s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_256,main="Correlation Dissimilarity",col=H_sin_cor_50s_2048_256_cl)
plt.suptitle("MDS of 50s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_256 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_256,main="Euclidean Distance",col=H_sin_eu_50s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_256,main="Manhattan Distance",col=H_sin_L1_50s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_256,main="Correlation Dissimilarity",col=H_sin_cor_50s_1024_256_cl)
plt.suptitle("MDS of 50s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_128 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_128,main="Euclidean Distance",col=H_sin_eu_50s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_128,main="Manhattan Distance",col=H_sin_L1_50s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_128,main="Correlation Dissimilarity",col=H_sin_cor_50s_1024_128_cl)
plt.suptitle("MDS of 50s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))


file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_512 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_512,main="Euclidean Distance",col=H_ave_eu_50s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_512,main="Manhattan Distance",col=H_ave_L1_50s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_512,main="Correlation Dissimilarity",col=H_ave_cor_50s_2048_512_cl)
plt.suptitle("MDS of 50s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_256 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_256,main="Euclidean Distance",col=H_ave_eu_50s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_256,main="Manhattan Distance",col=H_ave_L1_50s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_256,main="Correlation Dissimilarity",col=H_ave_cor_50s_2048_256_cl)
plt.suptitle("MDS of 50s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_256 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_256,main="Euclidean Distance",col=H_ave_eu_50s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_256,main="Manhattan Distance",col=H_ave_L1_50s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_256,main="Correlation Dissimilarity",col=H_ave_cor_50s_1024_256_cl)
plt.suptitle("MDS of 50s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_128 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_128,main="Euclidean Distance",col=H_ave_eu_50s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_128,main="Manhattan Distance",col=H_ave_L1_50s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_128,main="Correlation Dissimilarity",col=H_ave_cor_50s_1024_128_cl)
plt.suptitle("MDS of 50s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))



file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_512 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_512,main="Euclidean Distance",col=H_com_eu_20s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_512,main="Manhattan Distance",col=H_com_L1_20s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_512,main="Correlation Dissimilarity",col=H_com_cor_20s_2048_512_cl)
plt.suptitle("MDS of 20s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_256 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_256,main="Euclidean Distance",col=H_com_eu_20s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_256,main="Manhattan Distance",col=H_com_L1_20s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_256,main="Correlation Dissimilarity",col=H_com_cor_20s_2048_256_cl)
plt.suptitle("MDS of 20s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_256 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_256,main="Euclidean Distance",col=H_com_eu_20s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_256,main="Manhattan Distance",col=H_com_L1_20s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_256,main="Correlation Dissimilarity",col=H_com_cor_20s_1024_256_cl)
plt.suptitle("MDS of 20s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_128 Features Using Complete Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_128,main="Euclidean Distance",col=H_com_eu_20s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_128,main="Manhattan Distance",col=H_com_L1_20s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_128,main="Correlation Dissimilarity",col=H_com_cor_20s_1024_128_cl)
plt.suptitle("MDS of 20s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_512 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_512,main="Euclidean Distance",col=H_sin_eu_20s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_512,main="Manhattan Distance",col=H_sin_L1_20s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_512,main="Correlation Dissimilarity",col=H_sin_cor_20s_2048_512_cl)
plt.suptitle("MDS of 20s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_256 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_256,main="Euclidean Distance",col=H_sin_eu_20s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_256,main="Manhattan Distance",col=H_sin_L1_20s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_256,main="Correlation Dissimilarity",col=H_sin_cor_20s_2048_256_cl)
plt.suptitle("MDS of 20s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_256 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_256,main="Euclidean Distance",col=H_sin_eu_20s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_256,main="Manhattan Distance",col=H_sin_L1_20s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_256,main="Correlation Dissimilarity",col=H_sin_cor_20s_1024_256_cl)
plt.suptitle("MDS of 20s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_128 Features Using Single Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_128,main="Euclidean Distance",col=H_sin_eu_20s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_128,main="Manhattan Distance",col=H_sin_L1_20s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_128,main="Correlation Dissimilarity",col=H_sin_cor_20s_1024_128_cl)
plt.suptitle("MDS of 20s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))


file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_512 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_512,main="Euclidean Distance",col=H_ave_eu_20s_2048_512_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_512,main="Manhattan Distance",col=H_ave_L1_20s_2048_512_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_512,main="Correlation Dissimilarity",col=H_ave_cor_20s_2048_512_cl)
plt.suptitle("MDS of 20s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_256 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_256,main="Euclidean Distance",col=H_ave_eu_20s_2048_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_256,main="Manhattan Distance",col=H_ave_L1_20s_2048_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_256,main="Correlation Dissimilarity",col=H_ave_cor_20s_2048_256_cl)
plt.suptitle("MDS of 20s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_256 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_256,main="Euclidean Distance",col=H_ave_eu_20s_1024_256_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_256,main="Manhattan Distance",col=H_ave_L1_20s_1024_256_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_256,main="Correlation Dissimilarity",col=H_ave_cor_20s_1024_256_cl)
plt.suptitle("MDS of 20s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_128 Features Using Average Linkage.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_128,main="Euclidean Distance",col=H_ave_eu_20s_1024_128_cl)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_128,main="Manhattan Distance",col=H_ave_L1_20s_1024_128_cl)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_128,main="Correlation Dissimilarity",col=H_ave_cor_20s_1024_128_cl)
plt.suptitle("MDS of 20s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

adjusted_rand_score(lab,H_com_eu_50s_2048_512_cl)
adjusted_rand_score(lab,H_com_L1_50s_2048_512_cl)
adjusted_rand_score(lab,H_com_cor_50s_2048_512_cl)
adjusted_rand_score(lab,H_com_eu_50s_2048_256_cl)
adjusted_rand_score(lab,H_com_L1_50s_2048_256_cl)
adjusted_rand_score(lab,H_com_cor_50s_2048_256_cl)
adjusted_rand_score(lab,H_com_eu_50s_1024_256_cl)
adjusted_rand_score(lab,H_com_L1_50s_1024_256_cl)
adjusted_rand_score(lab,H_com_cor_50s_1024_256_cl)
adjusted_rand_score(lab,H_com_eu_50s_1024_128_cl)
adjusted_rand_score(lab,H_com_L1_50s_1024_128_cl)
adjusted_rand_score(lab,H_com_cor_50s_1024_128_cl)

adjusted_rand_score(lab,H_com_eu_20s_2048_512_cl)
adjusted_rand_score(lab,H_com_L1_20s_2048_512_cl)
adjusted_rand_score(lab,H_com_cor_20s_2048_512_cl)
adjusted_rand_score(lab,H_com_eu_20s_2048_256_cl)
adjusted_rand_score(lab,H_com_L1_20s_2048_256_cl)
adjusted_rand_score(lab,H_com_cor_20s_2048_256_cl)
adjusted_rand_score(lab,H_com_eu_20s_1024_256_cl)
adjusted_rand_score(lab,H_com_L1_20s_1024_256_cl)
adjusted_rand_score(lab,H_com_cor_20s_1024_256_cl)
adjusted_rand_score(lab,H_com_eu_20s_1024_128_cl)
adjusted_rand_score(lab,H_com_L1_20s_1024_128_cl)
adjusted_rand_score(lab,H_com_cor_20s_1024_128_cl)
#####PAM#####

def PAM(dist):
    best_silhoutte=-100
    best_i=1
    for i in range(2,10):
        km=KMedoids(n_clusters=i,metric="precomputed",method="pam")
        clusters=km.fit_predict(dist)
        silhouette=silhouette_score(dist,clusters,metric="precomputed")
        if silhouette>best_silhoutte:
            best_silhoutte=silhouette
            best_k=i
    best_km=KMedoids(n_clusters=best_k,metric="precomputed",method="pam")
    best_km.fit(dist)
    return best_km

PAM_eu_50s_2048_512=PAM(eu_dist_mfcc_50s_2048_512) #PAM_cor_20s_2048_512.labels_
PAM_eu_50s_2048_256=PAM(eu_dist_mfcc_50s_2048_256)
PAM_eu_50s_1024_256=PAM(eu_dist_mfcc_50s_1024_256)
PAM_eu_50s_1024_128=PAM(eu_dist_mfcc_50s_1024_128)

PAM_L1_50s_2048_512=PAM(L1_dist_mfcc_50s_2048_512) 
PAM_L1_50s_2048_256=PAM(L1_dist_mfcc_50s_2048_256)
PAM_L1_50s_1024_256=PAM(L1_dist_mfcc_50s_1024_256)
PAM_L1_50s_1024_128=PAM(L1_dist_mfcc_50s_1024_128)

PAM_cor_50s_2048_512=PAM(cor_dist_mfcc_50s_2048_512) 
PAM_cor_50s_2048_256=PAM(cor_dist_mfcc_50s_2048_256)
PAM_cor_50s_1024_256=PAM(cor_dist_mfcc_50s_1024_256)
PAM_cor_50s_1024_128=PAM(cor_dist_mfcc_50s_1024_128)

PAM_eu_20s_2048_512=PAM(eu_dist_mfcc_20s_2048_512) 
PAM_eu_20s_2048_256=PAM(eu_dist_mfcc_20s_2048_256)
PAM_eu_20s_1024_256=PAM(eu_dist_mfcc_20s_1024_256)
PAM_eu_20s_1024_128=PAM(eu_dist_mfcc_20s_1024_128)

PAM_L1_20s_2048_512=PAM(L1_dist_mfcc_20s_2048_512) 
PAM_L1_20s_2048_256=PAM(L1_dist_mfcc_20s_2048_256)
PAM_L1_20s_1024_256=PAM(L1_dist_mfcc_20s_1024_256)
PAM_L1_20s_1024_128=PAM(L1_dist_mfcc_20s_1024_128)

PAM_cor_20s_2048_512=PAM(cor_dist_mfcc_20s_2048_512) 
PAM_cor_20s_2048_256=PAM(cor_dist_mfcc_50s_2048_256)
PAM_cor_20s_1024_256=PAM(cor_dist_mfcc_50s_1024_256)
PAM_cor_20s_1024_128=PAM(cor_dist_mfcc_50s_1024_128)

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_512 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_512,main="Euclidean Distance",col=PAM_eu_50s_2048_512.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_512,main="Manhattan Distance",col=PAM_L1_50s_2048_512.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_512,main="Correlation Dissimilarity",col=PAM_cor_50s_2048_512.labels_)
plt.suptitle("MDS of 50s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_2048_256 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_2048_256,main="Euclidean Distance",col=PAM_eu_50s_2048_256.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_2048_256,main="Manhattan Distance",col=PAM_L1_50s_2048_256.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_2048_256,main="Correlation Dissimilarity",col=PAM_cor_50s_2048_256.labels_)
plt.suptitle("MDS of 50s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_256 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_256,main="Euclidean Distance",col=PAM_eu_50s_1024_256.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_256,main="Manhattan Distance",col=PAM_L1_50s_1024_256.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_256,main="Correlation Dissimilarity",col=PAM_cor_50s_1024_256.labels_)
plt.suptitle("MDS of 50s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 50s_1024_128 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_50s_1024_128,main="Euclidean Distance",col=PAM_eu_50s_1024_128.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_50s_1024_128,main="Manhattan Distance",col=PAM_L1_50s_1024_128.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_50s_1024_128,main="Correlation Dissimilarity",col=PAM_cor_50s_1024_128.labels_)
plt.suptitle("MDS of 50s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_512 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_512,main="Euclidean Distance",col=PAM_eu_20s_2048_512.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_512,main="Manhattan Distance",col=PAM_L1_20s_2048_512.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_512,main="Correlation Dissimilarity",col=PAM_cor_20s_2048_512.labels_)
plt.suptitle("MDS of 20s_2048_512 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_2048_256 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_2048_256,main="Euclidean Distance",col=PAM_eu_20s_2048_256.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_2048_256,main="Manhattan Distance",col=PAM_L1_20s_2048_256.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_2048_256,main="Correlation Dissimilarity",col=PAM_cor_20s_2048_256.labels_)
plt.suptitle("MDS of 20s_2048_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_256 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_256,main="Euclidean Distance",col=PAM_eu_20s_1024_256.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_256,main="Manhattan Distance",col=PAM_L1_20s_1024_256.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_256,main="Correlation Dissimilarity",col=PAM_cor_20s_1024_256.labels_)
plt.suptitle("MDS of 20s_1024_256 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

file_path=os.path.join(project_root,"Statistical_Sciences_template","figure","MDS of 20s_1024_128 Features Using PAM.png") 
plt.figure(figsize=(21, 12))
plt.subplot(1,3,1)
mds_visualization(dist_matrix=eu_dist_mfcc_20s_1024_128,main="Euclidean Distance",col=PAM_eu_20s_1024_128.labels_)
plt.subplot(1,3,2)
mds_visualization(dist_matrix=L1_dist_mfcc_20s_1024_128,main="Manhattan Distance",col=PAM_L1_20s_1024_128.labels_)
plt.subplot(1,3,3)
mds_visualization(dist_matrix=cor_dist_mfcc_20s_1024_128,main="Correlation Dissimilarity",col=PAM_cor_20s_1024_128.labels_)
plt.suptitle("MDS of 20s_1024_128 Features",fontsize=20,y=1.00001)
plt.subplots_adjust(left=0.1,right=0.9,bottom=0.1,top=0.9,wspace=0.4,hspace=0.4)
plt.savefig(os.path.join(file_path))

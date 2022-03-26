import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA, FactorAnalysis
from scipy.cluster.hierarchy import dendrogram, linkage
        
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("D:/Dersler/Yapay Zeka - Ecodation/Çalışmalar/1/CC GENERAL (1).csv")
veriler = df.copy()

veriler.columns = veriler.columns.str.lower()

veriler.head()

veriler.describe()

veriler.shape

veriler.info()


#https://www.geeksforgeeks.org/python-pandas-dataframe-fillna-to-replace-null-values-in-dataframe/
veriler.isnull().sum().sort_values(ascending=False)
veriler.dropna(inplace=True)

o_cols = veriler.select_dtypes(include=['object']).columns.tolist()
num_cols = veriler.select_dtypes(exclude=['object']).columns.tolist()

sns.pairplot(veriler)
plt.show()

plt.figure(figsize=(20,20))
corr_data = veriler.corr()
sns.heatmap(corr_data,annot=True)
plt.show()

scaler = StandardScaler()
data_scaled = scaler.fit_transform(veriler)
data_scaled.shape


wcss= []

for i in range(1,11): 
    km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=42)
    km.fit(data_scaled)
    wcss.append(km.inertia_)
    
plt.plot(range(1,11),wcss, marker='o', linestyle='--')
plt.title('Dirsek Metodu', fontsize =20)
plt.xlabel('Kümeler')
plt.ylabel('wcss')

plt.show()

km = KMeans(n_clusters=4, init='k-means++', n_init=10, max_iter=300, random_state=42)
label = km.fit_predict(data_scaled)

veriler['label'] = label

veriler['constant'] = 'constant'

plt.rcParams['figure.figsize'] =(25,40)

for num in range(0,17):
    ax = plt.subplot(5,4,num+1)
    col = veriler.columns[num]
    sns.stripplot(veriler['constant'],veriler[col], ax=ax, hue=veriler['label'])
    plt.xlabel('constant')

plt.show()

#PCA
pca = PCA(n_components = 7)  
pca.fit(data_scaled)
data_scaled.shape
x_pca = pca.transform(data_scaled)
x_pca.shape
print("variance ratio: ", pca.explained_variance_ratio_)
print("sum: ",sum(pca.explained_variance_ratio_))


#data["p1"] = x_pca[:,0]
#data["p2"] = x_pca[:,1]

x = x_pca[:,0]
y = x_pca[:,1]

colors = {0: 'red',
          1: 'blue',
          2: 'green', 
          3: 'yellow'}
  
veriler = pd.DataFrame({'x': x, 'y':y, 'label':veriler['label']}) 
groups = veriler.groupby('label')

fig, ax = plt.subplots(figsize=(20, 13)) 

for name, group in groups:
    ax.plot(group.x, group.y, marker='o', linestyle='', ms=5, color=colors[name], mec='none')
    ax.set_aspect('auto')
    ax.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
    ax.tick_params(axis= 'y',which='both',left='off',top='off',labelleft='off')
    
ax.set_title("Kredi Kartı kullanımına yönelik müşteri sınıflandırılması")
plt.show()



plt.figure(figsize=(20,20))
corr_pca = veriler.corr()
sns.heatmap(corr_pca,annot=True)
plt.show()
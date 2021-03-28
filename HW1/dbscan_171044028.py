###
# CSE454 Data Mining - Assignment 1
# Nurettin Cem Dedetas
# 171044028
###


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter
from sklearn.neighbors import NearestNeighbors



df = pd.read_csv("2d_dataset_8.csv") #dataset with various different cluster types



### Visualisation of raw data ###
_ = plt.plot(df['alpha'],df['beta'],marker='.',linewidth=0, color = '#229a33')

_ = plt.grid(which = 'major', color = '#cccccc', alpha = 0.45)

_ = plt.title('Distribution')

_ = plt.xlabel('x coordinates')
_ = plt.ylabel('y coordiantes')
_ = plt.show()




#converting raw data to workable numpy array
dbscan_data = df[['alpha','beta']]
dbscan_data = dbscan_data.values.astype('float32',copy=False)




### Nnormalised the data  ###
dbscan_data_scalar = StandardScaler().fit(dbscan_data)
dbscan_data = dbscan_data_scalar.transform(dbscan_data)



#Getting the K distance
nbrs = NearestNeighbors(n_neighbors=2).fit(dbscan_data)
distances, indices = nbrs.kneighbors(dbscan_data)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
#epsilon = distances[len(distances)-1]+distances[int((len(distances)/3))]  #for testing purposes



# Plotting K-distance Graph

plt.figure()
plt.plot(distances)
plt.title('K-Distance Graph',fontsize=16)
plt.xlabel('Distances between neighbouring data points',fontsize=12)
plt.ylabel('Epsilon',fontsize=12)
plt.show()



### Running the DBSCAN algorithm with acquired epsilon(radius) & minimum points data ###
model = DBSCAN(eps=0.07, min_samples=4,metric='euclidean', n_jobs = -1).fit(dbscan_data)



### Getting the processed info(number of clusters and outliers) ###
cls=0
clusters = Counter(model.labels_)
if(clusters[-1] >0):
    cls=1
    
print("There are {} clusters.".format(len(clusters)-cls))
print("There are {} outliers.".format(clusters[-1]))
print("Outlier percentage(purity) is %{}  .".format(round(clusters[-1]/len(df),2)))



### Making different clusters unique colors###
outl_df= df[model.labels_ == -1]
clusters_df = df[model.labels_ != -1]
colors = model.labels_
colors_clusters = colors[colors != -1]
colors_outliers = 'black'




### visualise the results###
fig = plt.figure()

ax = fig.add_axes([.1,.1,1,1])

ax.scatter(clusters_df['alpha'], clusters_df['beta'],c= colors_clusters,edgecolors = 'grey', s=28)
ax.scatter(outl_df['alpha'], outl_df['beta'],c= colors_outliers,edgecolors = 'red', s=17)

ax.set_xlabel('Y')
ax.set_ylabel('X')

plt.title('Distribution')
plt.grid(which='major', color = '#cccccc', alpha = 0.5)
plt.show()


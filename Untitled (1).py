#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px


# In[8]:


spotify=pd.read_csv("spotify dataset.csv")


# In[9]:


spotify


# In[10]:


spotify.head(10)


# In[11]:


spotify.isnull().sum()


# In[12]:


spotify.dropna(inplace=True)


# In[13]:


spotify.isnull().sum()


# In[115]:


spotify.duplicated().sum()


# In[116]:


sns.set(style="ticks", context="talk")
plt.style.use("dark_background")


# In[9]:


sns.pairplot(spotify,corner=True,hue='playlist_genre')


# In[117]:


# set up size and color for sns
sns.set(rc={'figure.figsize':(15,4)})
plt.rcParams['figure.dpi'] = 300
plt.style.use('fivethirtyeight')


# In[118]:


# top 10 popular songs in the dataset
songs = spotify.groupby('track_name')['track_popularity'].mean().sort_values(ascending=False)[:10]
sns.barplot(x=songs, y=songs.index, orient = 'h')
plt.xlabel('track_popularity', fontsize=14)
plt.ylabel('track_name', fontsize=14)
plt.tight_layout()


# # Let's have a look at the average time of a song

# In[14]:


px.box(data_frame=spotify,y='duration_ms',color='playlist_genre')


# # HeatMap of the data

# In[120]:


corr_matrix=spotify.corr()


# In[121]:


features=corr_matrix.index
features


# In[122]:


plt.figure(figsize=(20,20))
sns.heatmap(spotify[features].corr(), annot = True)


# In[257]:


plt.scatter(spotify[["energy"]],spotify[["loudness"]])


# In[240]:


from sklearn.cluster import KMeans
kmeans=KMeans(n_clusters=6)
kmeans.fit(spotify[["energy","loudness"]])


# In[241]:


kmeans.cluster_centers_


# In[242]:


kmeans.labels_


# In[243]:


spotify["Cluster Group"]=kmeans.labels_


# In[244]:


spotify


# In[245]:


spotify["Cluster Group"].value_counts()


# In[246]:


sns.scatterplot(x="energy",y="loudness",data=spotify,hue="Cluster Group")


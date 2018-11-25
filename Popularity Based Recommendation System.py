# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 00:35:43 2018

@author: MAC
"""
import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np
import time
from sklearn.externals import joblib
import Recommenders as Recommenders

triplets_file='https://static.turi.com/datasets/millionsong/10000.txt'
songs_metadata_file='https://static.turi.com/datasets/millionsong/song_data.csv'

song_df_1=pd.read_table(triplets_file,header=None)
song_df_1.columns=['user_id','song_id','listen_count']

song_df_2 = pd.read_csv(songs_metadata_file)
song_df = pd.merge(song_df_1,song_df_2.drop_duplicates(['song_id']),on="song_id",how="left")

song_grouped = song_df.groupby(['title']).agg({'listen_count':'count'}).reset_index()
grouped_sum = song_grouped['listen_count'].sum()
song_grouped['percentage'] = song_grouped['listen_count'].div(grouped_sum)*100
song_grouped.sort_values(['listen_count','title'],ascending=[0,1])

users = song_df['user_id'].unique()
len(users)
songs = song_df['title'].unique()
len(songs)

train_data,test_data=train_test_split(song_df,test_size=0.20,random_state=0)
 
"""
Popularity Based Sysem
"""

train_data_grouped = train_data.groupby(['song_id']).agg({'user_id' : 'count'}).reset_index()
train_data_grouped.rename(columns = {'user_id' : 'score'},inplace=True)
        
train_data_sort = train_data_grouped.sort_values(['score','song_id'],ascending = [0,1])
        
train_data_sort['Rank'] = train_data_sort['score'].rank(ascending=0,method='first')
        
popularity_recommendations = train_data_sort.head(10)

user_id=users[5]

user_recommendation = popularity_recommendations
user_recommendation['user_id'] = user_id
cols = user_recommendation.columns.tolist()
cols = cols[-1:] + cols[:-1]
user_recommendation = user_recommendation[cols]
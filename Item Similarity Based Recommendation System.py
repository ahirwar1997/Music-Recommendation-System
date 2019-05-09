# -*- coding: utf-8 -*-
"""
Created on Mon Dec  3 15:44:17 2018

@author: MAC
"""

import pandas as pd
from sklearn.cross_validation import train_test_split
import numpy as np

"""
    Data extraction and transformation
"""
#Storing songs data and users data links
everyuserdata = 'https://static.turi.com/datasets/millionsong/10000.txt'
everysongdata = 'https://static.turi.com/datasets/millionsong/song_data.csv'

#Storing users data
dataframe_1=pd.read_table(everyuserdata,header=None)
dataframe_1.columns=['user_id','song_id','listen_count']

#Storing songs data
dataframe_2 = pd.read_csv(everysongdata)

#Joining users and songs

song_dataframe = pd.merge(dataframe_1,dataframe_2.drop_duplicates(['song_id']),on="song_id",how="left")

#Sorting songs according to their listen count percentage
song_percentage = song_dataframe.groupby(['title']).agg({'listen_count':'count'}).reset_index()
grouped_sum = song_percentage['listen_count'].sum()
song_percentage['percentage'] = song_percentage['listen_count'].div(grouped_sum)*100
song_percentage.sort_values(['listen_count','title'],ascending=[0,1])

#list of all unique users
users = song_dataframe['user_id'].unique()

#list of all unique songs
songs = song_dataframe['title'].unique()

#creating train_data(80%)and test_data(20%)
train_data,test_data=train_test_split(song_dataframe,test_size=0.20,random_state=0)

"""
    User Similarity Based Recommendation System
"""
#Selecting a user to recommend for
user_id = users[5]

#Getting the user's data and song_ids of all listened songs
user_data = train_data.loc[train_data['user_id'] == user_id]
user_songs = list(user_data['song_id'].unique())

#Storing all songs ids
all_songs = list(train_data['song_id'].unique())

#Creating a cooccurence matrix (line 58 - 90)

#creating array containing sets of users listening to training user's each songs
user_songs_user = []
for i in range(0,len(user_songs)):
    #user_songs_user.append(get_item_users(user[i]))
    item_data = train_data.loc[train_data['song_id']==user_songs[i]]
    item_users = set(item_data['user_id'].unique())
    user_songs_user.append(item_users)

#creating a cooccurence 2d matrix    
cooccurence_matrix = np.matrix(np.zeros(shape=(len(user_songs),len(all_songs))),float)
for i in range(0,len(all_songs)):
    #Calculate unique listeners of song i
    songs_i_data = train_data.loc[train_data['song_id'] == all_songs[i]]
    users_i = set(songs_i_data['user_id'].unique())
    
    for j in range(0,len(user_songs)):
        #Get unique listeners of song j
        users_j = user_songs_user[j]
        
        #Calculate intersection of listens of song i and j
        users_intersect = users_i.intersection(users_j)
        
        #Calculate cooccurence matrix[i,j] as Jaccard Index
        if len(users_intersect) != 0:
            
            #Calculating union of songs i and j
            users_union = users_i.union(users_j)
            cooccurence_matrix[j,i] = float(len(users_intersect))/float(len(users_union))
       
        else:
            cooccurence_matrix[j,i] = 0;

#Making recommendations

#Calculate a weighted average of the scores in cooccurence matrix for all user songs.
user_song_scores = cooccurence_matrix.sum(axis = 0)/float(cooccurence_matrix.shape[0])
user_song_scores = np.array(user_song_scores)[0].tolist()

#Sort the indices of user_sim_scores based upon their value plus maintain the corresponding score
sort_index = sorted(((e,i) for i,e in enumerate(list(user_song_scores))),reverse = True)

#Create a dataframe for result recommendations
columns = ['user_id','song','rank','score']
recommendations = pd.DataFrame(columns=columns)
 
#Fill the dataframe with top 10 recommendations
start = 1 
for i in range(0,len(sort_index)):
    if  start <=10 and ~np.isnan(sort_index[i][0]) and all_songs[sort_index[i][1]] not in user_songs:
        x = train_data.loc[train_data['song_id'] == all_songs[sort_index[i][1]]]
        y= x['title'].tolist()
        recommendations.loc[len(recommendations)] = [user_id,y[0],start,sort_index[i][0]]
        start = start + 1
if recommendations.shape[0] == 0:
    print("No recommendations!")
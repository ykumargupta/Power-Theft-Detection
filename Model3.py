#!/usr/bin/env python
# coding: utf-8

# ## Accuracy Score of ~ 76%
# 
# ## Machine Learning Algorithm For Efficient Power Theft Detection Using Smart Meter Data

# ## Dataset - The Smart Meter dataset is provided by Irish Social Science Data Archive‟s (ISSDA), Ireland. The dataset used for the experiment is the subset of smart meter dataset of Ireland in December 2010. The data set considered in this work is residential smart meter data. The data contains the information about the customer id, code for date/time, electricity consumption for every 30 minutes (in KWh).

# ### We like to acknowledge with thanks to Irish Social Science Data Archive Center for providing us with the access to the Smart Meter Data Archive that had a vital role for this research work. 

# In[14]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras


# # Data Preprocessing

# In[15]:


columns = ['id','daytime','power']
data1 = pd.read_csv('File1.txt',sep = ' ',names = columns)
data1['day']=data1['daytime'][:]
data1['time'] = data1['daytime'][:]
data1['day'] = data1['day']/100
data1['day'] = np.int64(data1['day'])
data1['time']= data1['time']%100
data1['time'] = np.int64(data1['time'])
data1=data1.drop(columns= ['daytime'])
data1 = data1[(data1.day<200) & (data1.day>=195)]


# In[16]:


from collections import Counter
count = 0
temp = data1['id'].values.tolist()
no_of_occur = Counter(temp)


# In[17]:


data1.shape


# In[18]:


data1.head()


# In[19]:


no_of_occur


# In[20]:


new_data = data1.drop([1489],axis = 0)


# In[21]:


new_data.head()


# In[22]:


new_data.reset_index(drop=True)


# In[23]:


new_data.ix[:,2] -= 194


# In[24]:


new_data


# In[25]:


new_data.values[:,0].tolist().count(1392)


# In[26]:


new_data.ix[:,0]


# In[27]:


customers = {}
count = 0
for ix in range(new_data.shape[0]):
    if new_data['id'].iloc[ix] not in customers.keys():
        customers[new_data['id'].iloc[ix]] = new_data['power'].iloc[ix]
    else:
        temp = {}
        temp[new_data['id'].iloc[ix]] = new_data['power'].iloc[ix]+ customers[new_data['id'].iloc[ix]]
        customers.update(temp)
    count +=1
    print (count)


# In[28]:


customers


# In[29]:


len(customers.keys())


# In[30]:


data_new = pd.DataFrame(customers.items(), columns=['id', 'totalpower'])


# In[31]:


data_new = data_new.sort_values('id')


# In[32]:


data_new.head()


# In[33]:


data_new.shape


# ## K Means Clustering

# In[34]:


from sklearn.cluster import KMeans


# In[35]:


wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(data_new)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# ## Optimum No of Clusters - 4  ( By Elbow Method)

# In[36]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42)
y_kmeans = kmeans.fit_predict(data_new)


# In[37]:


y_kmeans


# In[38]:


unique_elements, counts_elements = np.unique(y_kmeans, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))


# In[39]:


type(y_kmeans)


# In[40]:


len(y_kmeans)


# In[41]:


type(data_new['id'])


# In[42]:


new_data = new_data.sort_values('id')


# In[43]:


gen_data = pd.DataFrame(columns= ['id','power','day'])
count =0
for ix in range(len(y_kmeans)):
    if y_kmeans[ix] == 1:
        for iy in range(new_data.shape[0]):
            if data_new['id'].iloc[ix] != new_data['id'].iloc[iy] :
                pass
            else:
                gen_data.loc[count] = [new_data['id'].iloc[iy],new_data['power'].iloc[iy],new_data['day'].iloc[iy]]
#        gen_data.loc[count] = [data_new['id'].iloc[ix],new_data['power'].iloc[ix],new_data['day'].iloc[ix]]
                count +=1
                print(count)
            
    


# In[44]:


gen_data


# In[45]:


df = pd.concat([gen_data,pd.get_dummies(gen_data['day'], prefix='day')],axis=1)

# now drop the original 'country' column (you don't need it anymore)
df.drop(['day'],axis=1, inplace=True)


# In[46]:


df


# In[47]:


for ix in range(df.shape[0]):
    df['day_1.0'].loc[ix] = df['day_1.0'].loc[ix]*df['power'].loc[ix]
    df['day_2.0'].loc[ix] = df['day_2.0'].loc[ix]*df['power'].loc[ix]
    df['day_3.0'].loc[ix] = df['day_3.0'].loc[ix]*df['power'].loc[ix]
    df['day_4.0'].loc[ix] = df['day_4.0'].loc[ix]*df['power'].loc[ix]
    df['day_5.0'].loc[ix] = df['day_5.0'].loc[ix]*df['power'].loc[ix]


# In[48]:


df.head()


# In[49]:


kk = {}
count = 0
for ix in range(df.shape[0]):
    if df.ix[ix,0] not in kk.keys():
        kk[df.ix[ix,0]] = {
            "day_1.0" : df.ix[ix,2],
            "day_2.0" : df.ix[ix,3],
            "day_3.0" : df.ix[ix,4],
            "day_4.0" : df.ix[ix,5],
            "day_5.0" : df.ix[ix,6],
        }
#        [df.ix[ix,2],df.ix[ix,3],df.ix[ix,4],df.ix[ix,5],df.ix[ix,6]]
    else:
        temp = {}
        temp[df.ix[ix,0]] = {
            "day_1.0" : df.ix[ix,2]+kk[df.ix[ix,0]]["day_1.0"],
            "day_2.0" : df.ix[ix,3]+kk[df.ix[ix,0]]["day_2.0"],
            "day_3.0" : df.ix[ix,4]+kk[df.ix[ix,0]]["day_3.0"],
            "day_4.0" : df.ix[ix,5]+kk[df.ix[ix,0]]["day_4.0"],
            "day_5.0" : df.ix[ix,6]+kk[df.ix[ix,0]]["day_5.0"],
        }
#        [df.ix[ix,2]+kk[df.ix[ix,0]],df.ix[ix,3]+kk[df.ix[ix,0]],df.ix[ix,4]+kk[df.ix[ix,0]],df.ix[ix,5]+kk[df.ix[ix,0]],df.ix[ix,6]+kk[df.ix[ix,0]]]
        kk.update(temp)
    count+= 1
    print (count)


# In[50]:


kk.keys()


# In[51]:


kk_new = pd.DataFrame(kk, columns=['id', 'day_1.0','day_2.0','day_3.0','day_4.0','day_5.0'])


# In[52]:


kk.items()
data_new = pd.DataFrame(customers.items(), columns=['id', 'totalpower'])


# In[53]:


df2 = pd.DataFrame( [[i,kk[i]['day_1.0'],kk[i]['day_2.0'],kk[i]['day_3.0'],kk[i]['day_4.0'],kk[i]['day_5.0'] ] for i in kk.keys()] , columns = ['id','day_1.0','day_2.0','day_3.0','day_4.0','day_5.0'])


# In[54]:


df2


# In[55]:


df3 = df2


# In[57]:


import random
multi = random.random()


# # Bogus Data

# In[58]:



for ix in range(df3.shape[0]):
    for iy in range(df3.shape[1]-1):
        df3.ix[ix,iy+1] = (df3.ix[ix,iy+1]*multi)/48


# In[59]:


df3 


# In[60]:


df4 = pd.DataFrame( [[i,kk[i]['day_1.0'],kk[i]['day_2.0'],kk[i]['day_3.0'],kk[i]['day_4.0'],kk[i]['day_5.0'] ] for i in kk.keys()] , columns = ['id','day_1.0','day_2.0','day_3.0','day_4.0','day_5.0'])


# In[61]:


df4


# In[62]:


df1_elements = df4.sample(n=32)
type(df1_elements)


# In[63]:


df1_elements


# In[64]:


for ix in range(df1_elements.shape[0]):
    gg = df1_elements['id'].iloc[ix]
    for iy in range(df4.shape[0]):
        if df4['id'].iloc[iy] == gg:
            vv = (ix+1)%6
            if vv == 0:
                vv +=1
            col = df4.columns[vv]
            df4[col].iloc[ix] = 0


# In[65]:


df4


# In[66]:


df5 = pd.DataFrame( [[i,kk[i]['day_1.0'],kk[i]['day_2.0'],kk[i]['day_3.0'],kk[i]['day_4.0'],kk[i]['day_5.0'] ] for i in kk.keys()] , columns = ['id','day_1.0','day_2.0','day_3.0','day_4.0','day_5.0'])


# In[67]:


df5


# In[68]:


mean = []

for ix in range(df5.shape[0]):
    sum = 0
    for iy in range(df5.shape[1]-1):
        
        sum += df5.ix[ix,iy+1] 
    mean.append(sum/5)


# In[69]:


for ix in range(df5.shape[0]):
    
    for iy in range(df5.shape[1]-1):
        
        df5.ix[ix,iy+1] *= mean[ix]
    


# In[70]:


df5


# In[71]:


orig = pd.DataFrame( [[i,kk[i]['day_1.0'],kk[i]['day_2.0'],kk[i]['day_3.0'],kk[i]['day_4.0'],kk[i]['day_5.0'] ] for i in kk.keys()] , columns = ['id','day_1.0','day_2.0','day_3.0','day_4.0','day_5.0'])


# In[72]:


#orig  orig
#type1 df3
#type2 df4
#type3 df5
orig['label'] = 1
df3['label']=0
df4['label']=0
df5['label']=0

semifinal = pd.concat([orig,df3,df4,df5],ignore_index=True)


# In[73]:


final = semifinal.drop("id", axis=1)


# In[127]:


final


# In[177]:


divide = int(final.shape[0]*0.6)


# In[129]:


#np.random.shuffle(final)


# In[76]:


final.shape


# In[77]:


final.ix[0,0]


# In[116]:


X_train = final.ix[:divide,:-1]
y_train = final.ix[:divide,-1]
X_test = final.ix[divide:,:-1]
y_test = final.ix[divide:,-1]


# In[113]:


print (X_train.shape)
X_train.head()


# In[178]:


final2 = final.values


# In[184]:


np.random.shuffle(final2)


# In[185]:


X_train = final2[:divide,:-1]
y_train = final2[:divide,-1]
X_test = final2[divide:,:-1]
y_test = final2[divide:,-1]


# In[186]:


y_test


# In[80]:


len(kk.keys())


# In[ ]:


import random
multi = random.random()


# In[187]:


print (X_train.shape)
print (X_test.shape)


# # Neural Network Classification

# In[169]:


import keras
from keras.models import Sequential
from keras.layers import Dense


# In[170]:


from keras.callbacks import ModelCheckpoint


# In[188]:


print (X_train.shape)
print (X_test.shape)


# In[189]:


classifier2 = Sequential()


# In[190]:




classifier2.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu', input_dim =5))
classifier2.add(Dense(output_dim = 16, init = 'uniform', activation = 'relu'))
classifier2.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))

classifier2.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

classifier2.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

checkpointer2 = ModelCheckpoint(filepath='weights.hdf5', verbose=1, save_best_only=True)
classifier2.summary()
classifier2.fit(X_train, y_train, batch_size=32, epochs=100,validation_data=(X_test, y_test), callbacks=[checkpointer2])

y_pred = classifier2.predict(X_test)
y_pred = (y_pred > 0.5)
    
    


# In[192]:


classifier2.load_weights('weights.hdf5')


# In[205]:


y_pred = classifier2.predict(X_test)
y_pred = (y_pred > 0.5)


# In[196]:


np.sum(y_pred==y_test)/y_test.shape[0]


# In[199]:


y_test= np.reshape(y_test,(y_test.shape[0],1))


# In[204]:


y_pred


# In[206]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# In[207]:


cm


# In[208]:


from sklearn.metrics import accuracy_score


# In[209]:


print (accuracy_score(y_test,y_pred))


# ## Accuracy Score of ~ 76%

# 

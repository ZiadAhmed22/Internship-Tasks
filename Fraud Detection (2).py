#!/usr/bin/env python
# coding: utf-8

# # Ziad Ahmed Kamel

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


# In[2]:


df=pd.read_csv("PS_20174392719_1491204439457_log.csv")
df.head()


# In[3]:


df.info()


# In[4]:


print(df.duplicated().sum())


# In[5]:


df.isnull().sum()


# In[6]:


fraud_check = pd.value_counts(df['isFraud'], sort =True)
fraud_check.plot(kind = 'bar', rot=0, color='r')
plt.title("Normal and fraud distribution")
plt.xlabel("Class")
plt.ylabel("Frequency")
Labels = ['Normal','Fraud']
plt.xticks(range(2), Labels)
plt.show()


# In[7]:


fraud_people = df[df['isFraud']==1]
normal_people = df[df['isFraud']==0]


# In[8]:


fraud_people.shape


# In[9]:


normal_people.shape


# In[10]:


fraud_people['amount'].describe()


# In[11]:


normal_people['amount'].describe()


# In[12]:


df.corr
plt.figure(figsize =(20,20))
g=sns.heatmap(df.corr(),annot=True)


# In[13]:


columns = df.columns.tolist()
# Making our independent Features
columns = [var for var in columns if var not in ["isFraud"]]
# Making our Dependent Variable
target = "isFraud"
x = df[columns]
y = df[target]


# In[14]:


x['type'].replace(['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER'],['1','2','3','4','5',], inplace=True)


# In[15]:


x.drop(['nameOrig','nameDest','isFlaggedFraud'],axis=1, inplace=True)
x.head()


# In[16]:


y.head()


# In[17]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state = 42)


# In[18]:


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()


# In[19]:


knn.fit(x_train,y_train)


# In[21]:


from sklearn.metrics import confusion_matrix , accuracy_score , precision_score
y_predictKNN=knn.predict(x_test)
print(confusion_matrix(y_test,y_predictKNN))
print('Accuracy KNN :{0:.3f}'.format(accuracy_score(y_test,y_predictKNN)))
print('Precision KNN :{0:.3f}'.format(precision_score(y_test,y_predictKNN)))


# In[ ]:





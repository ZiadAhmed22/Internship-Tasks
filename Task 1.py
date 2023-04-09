#!/usr/bin/env python
# coding: utf-8
#predict the percentage of an student based on the no. of study hours
# In[172]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
import seaborn as sns

from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error


# In[173]:


df=pd.read_csv("https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv")
df.head()


# In[174]:


X = df.iloc[:, :-1].values  
Y = df.iloc[:, 1].values  


# In[175]:


df.isnull==True


# In[176]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, 
                            test_size=0.2, random_state=0) 


# In[177]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[178]:


plt.scatter(X, Y)
sns.regplot(x= df['Hours'], y= df['Scores'])
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')
plt.show()


# In[179]:


y_pred = model.predict(X_test)
y_pred


# In[180]:


df = pd.DataFrame({'Actual': Y_test, 'Predicted': y_pred})  
df 


# In[181]:


hours = [9.25]

result = model.predict([hours])
result


# In[182]:


print("Predicted Score = {}".format(round(result[0],2)))


# In[184]:


print('Mean Absolute Error:', 
      mean_absolute_error(Y_test, y_pred)) 


# In[ ]:





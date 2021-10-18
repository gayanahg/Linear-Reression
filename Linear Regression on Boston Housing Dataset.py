#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
get_ipython().run_line_magic('matplotlib', 'inline')
dataset=datasets.load_boston()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
df.head(10)


# In[3]:


df.info()


# In[4]:


plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=True)
plt.show()


# In[7]:


for i, feature_name in enumerate(dataset.feature_names):
    plt.figure(figsize=(8, 5))
    plt.scatter(dataset.data[:, i], dataset.target)
    plt.ylabel('PRICE', size=10)
    plt.xlabel(feature_name, size=10)
    plt.show()


# In[8]:


x=dataset.data
y=dataset.target
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=32)


# In[10]:


log_regressor=LinearRegression()
log_regressor.fit(x_train,y_train)


# In[11]:


y_pred=log_regressor.predict(x_test)
print(y_pred)


# In[12]:


from sklearn.metrics import r2_score
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error : ", mse)
print("R squared error ", r2_score(y_test,y_pred))
print("root mean square error ",np.sqrt(mean_squared_error(y_test, y_pred)))
print("mean absolute error ",mean_absolute_error(y_test,y_pred))


# In[13]:


#Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor()
dtr.fit(x_train,y_train)


# In[14]:


y_pred=dtr.predict(x_test)
print(y_pred)


# In[15]:


from sklearn.metrics import r2_score
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error : ", mse)
print("R squared error ", r2_score(y_test,y_pred))
print("root mean square error ",np.sqrt(mean_squared_error(y_test, y_pred)))
print("mean absolute error ",mean_absolute_error(y_test,y_pred))


# In[ ]:





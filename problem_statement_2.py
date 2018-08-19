
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import sklearn
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score


# In[2]:


boston = load_boston()
bos = pd.DataFrame(boston.data)


# In[3]:


bos.head(10)


# In[4]:


target = pd.DataFrame(boston.target)


# In[5]:


target.head(10)


# In[8]:


X_train,X_test,y_train,y_test = train_test_split(bos,target,test_size=0.3,random_state=42)


# In[9]:


regressor = LinearRegression()


# In[10]:


regressor.fit(X_train,y_train)


# In[11]:


boston_pred = regressor.predict(X_test)


# In[12]:


print("mse = %.2f" %(mean_squared_error(boston_pred,y_test)))


# In[13]:


print("variance = %.2f" %(r2_score(boston_pred,y_test)))


# # Ideally, the scatter plot should create a linear line. Since the model does not fit 100%, the scatter plot is not creating a linear line.

# In[27]:


plt.scatter(y_test, boston_pred)
plt.xlabel("Prices: $Y_i$")
plt.ylabel("Predicted prices: $\hat{Y}_i$")
plt.title("Prices vs Predicted prices: $Y_i$ vs $\hat{Y}_i$")
plt.show()


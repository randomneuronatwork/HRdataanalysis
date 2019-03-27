#!/usr/bin/env python
# coding: utf-8

# In[249]:


import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix
import matplotlib.pyplot as plt
from sklearn.externals.six import StringIO
import pydotplus
from IPython.display import Image
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter
from xgboost import XGBClassifier
from xgboost import plot_tree
from sklearn import svm
import graphviz
import matplotlib as matplot
import seaborn as sns
get_ipython().magic(u'matplotlib inline')


# In[250]:


df=pd.read_csv("C:/Users/steve/Desktop/hr data/b.csv")
print(df)


# In[251]:


df.isnull().any()


# In[252]:


df.shape


# In[253]:


attrition_rate = df.Attrition.value_counts() / len(df)
print(attrition_rate)


# In[254]:


df.describe()


# In[255]:


attrition_Summary = df.groupby('Attrition')
attrition_Summary.mean()


# In[256]:


df.corr()


# In[257]:


emp_population = df['OverTime'][df['Attrition'] == 40].mean()
emp_turnover_Overtime = df[df['Attrition']==60]['OverTime'].mean()
print( 'The mean overtime for the employee population with no attrition is: ' + str(emp_population))
print( 'The mean overtime for employees that had a attrition is: ' + str(emp_turnover_Overtime) )


# In[258]:


print("Attrition count -        40 = No  60 = Yes")
print( df.Attrition.value_counts())
print("No. of people doing ovetime -        0 = No  10 = Yes")
print( df.OverTime.value_counts())


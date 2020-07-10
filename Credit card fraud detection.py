
# coding: utf-8

# In[5]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


df=pd.read_csv("creditcard.csv")


# In[7]:


df.info()


# In[8]:


df.head()


# In[9]:



sns.countplot(x='Class',data=df)


# In[10]:


#sns.pairplot(df.sample(1000))


# In[11]:


sns.heatmap(df.corr())


# In[12]:


df.drop('Time',axis=1,inplace=True)


# In[13]:


df.head(1)


# In[14]:


from sklearn.model_selection import train_test_split


# In[15]:


X=df.drop('Class',axis=1)
y=df['Class']


# In[16]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)


# In[17]:


from sklearn.linear_model import LinearRegression


# In[18]:


lm = LinearRegression()


# In[19]:


lm.fit(X_train,y_train)


# In[20]:


predictions = lm.predict(X_test)


# In[21]:


plt.scatter(y_test,predictions)


# In[22]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix


# In[23]:


print(classification_report(y_test,predictions))


# In[24]:


from sklearn.linear_model import LogisticRegression


# In[25]:


logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


# In[26]:


predictions = logmodel.predict(X_test)


# In[27]:


print(classification_report(y_test,predictions))


# In[28]:


print(confusion_matrix(y_test,predictions))


# In[29]:


from sklearn.neighbors import KNeighborsClassifier


# In[30]:


knn = KNeighborsClassifier(n_neighbors=1)


# In[31]:


knn.fit(X_train,y_train)


# In[32]:


pred = knn.predict(X_test)


# In[33]:


print(classification_report(y_test,pred))


# In[34]:


print(confusion_matrix(y_test,pred))


# In[ ]:


error_rate = []

# Will take some time
for i in range(1,10):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[35]:


from sklearn.tree import DecisionTreeClassifier


# In[36]:


dtree = DecisionTreeClassifier()


# In[38]:


dtree.fit(X_train,y_train)


# In[39]:


predictions = dtree.predict(X_test)


# In[40]:


print(classification_report(y_test,predictions))


# In[41]:


print(confusion_matrix(y_test,predictions))


# In[42]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)


# In[43]:


rfc_pred = rfc.predict(X_test)


# In[44]:


print(confusion_matrix(y_test,rfc_pred))


# In[45]:


print(classification_report(y_test,rfc_pred))


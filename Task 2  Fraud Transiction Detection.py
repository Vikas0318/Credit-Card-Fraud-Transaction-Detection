#!/usr/bin/env python
# coding: utf-8

# ## Task : Fraud Transaction Detection

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[2]:


# Loading the dataset to a pandas dataframe


# In[3]:


credit_card_data = pd.read_csv("C:\\Users\\Vikas\\OneDrive\\Desktop\\DATASET\\TechnoHacks Edutech\\creditcard.csv")


# In[4]:


credit_card_data


# In[5]:


credit_card_data.head()


# In[6]:


# Dataset information


# In[7]:


credit_card_data.info()


# In[8]:


# To check missing values 


# In[9]:


credit_card_data.isnull().sum()


# In[10]:


# Distribution of legit transaction and fraudulent transactions


# In[11]:


credit_card_data['Class'].value_counts()


# This dataset is highly unbalanced
0 --> Normal Transaction
1 --> Fraudulent Transaction  
# In[12]:


# Separating the data for analysis
legit = credit_card_data[credit_card_data.Class == 0]
fraud = credit_card_data[credit_card_data.Class == 1]


# In[13]:


print(legit.shape)
print(fraud.shape)


# In[14]:


# Statistical meansures of the data

legit.Amount.describe()


# In[15]:


fraud.Amount.describe()


# In[16]:


# Compare the values for both transaction

credit_card_data.groupby('Class').mean()


# Under - Samplig Method

# Build a sample dataset caontainng similar distribution of normal trasactions and Fraudulent Transactions

# Number of fraudulent transactions --> 492

# In[17]:


legit_sample  = legit.sample(n=492)


# Concatenating two DataFrames

# In[18]:


new_dataset = pd.concat([legit_sample,fraud],axis=0)
new_dataset.head()


# In[19]:


new_dataset.tail()


# In[20]:


new_dataset['Class'].value_counts()


# In[21]:


new_dataset.groupby('Class').mean()


# Splitting the data into Features and Targets

# In[22]:


X = new_dataset.drop(columns='Class',axis=1)
Y = new_dataset['Class']


# In[23]:


print(X)


# In[24]:


print(Y)


# Split the data into Training data and Testing data

# In[25]:


X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, stratify=Y, random_state=2)


# In[26]:


print(X.shape, X_train.shape, X_test.shape)


# Model Training

# Logistic Regression

# In[27]:


model = LogisticRegression()


# In[28]:


# Training the Logistic Regression Model with Training Data


# In[29]:


model.fit(X_train,Y_train)


# Model Evaluation

# Accuracy Score

# In[30]:


# Accuracy on training data
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)


# In[31]:


print('Accuracy on Training data : ', training_data_accuracy)


# In[32]:


# accuracy on test data


# In[33]:


X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction, Y_test)


# In[34]:


print('Accuracy on Testing data :', testing_data_accuracy)


#!/usr/bin/env python
# coding: utf-8

# # Author: Ayush Kakar
# 
# # Topic: Machine Learning " Hello World" Task 
#  # Dataset: "Titanic- Machine Learning from Disaster"
# 

# # Importing the necessary libraries

# In[275]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')


# In[276]:


from sklearn.ensemble import (RandomForestClassifier,
                             AdaBoostClassifier,
                             GradientBoostingClassifier)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (GridSearchCV,
                                     cross_val_score,
                                     StratifiedKFold, 
                                     learning_curve)


# ## Loading the dataset

# In[277]:


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
ids = test['PassengerId']
print('Train shape : ',train.shape)
print('Test shape : ',test.shape)


# In[278]:


train.head()


# In[279]:


test.head()


# ## Cleaning and preprocessing the dataset

# In[280]:


train.isna().sum().sort_values(ascending=False)


# In[281]:


train ['Embarked'].value_counts()


# In[282]:


train['Embarked'].fillna('S',inplace=True)
train.isna().sum().sort_values(ascending=False)


# In[283]:


sns.scatterplot(x='Age',y='SibSp',data=train)


# In[284]:


sns.scatterplot(x='Age',y='Parch',data=train)


# In[285]:


train[['Age','SibSp']].groupby('SibSp').median()


# In[286]:


train[['Age','SibSp']].groupby('SibSp').mean()


# In[287]:


train[['Age','Parch']].groupby('Parch').median()


# In[288]:


train[['Age','Parch']].groupby('Parch').mean()


# In[289]:


train[train['SibSp']==8]


# In[290]:


print('Mean of age is : ',train['Age'].mean())
print('Median of age is : ',train['Age'].median())


# In[291]:


train['Age'].fillna(train['Age'].median(),inplace=True)


# In[292]:


train.isna().sum().sort_values(ascending=False)


# # lets check the heatmap before decided about the cabin column
# 

# In[293]:


sns.heatmap(train.corr(),annot=True)


# # Similarly, lets map the sex column

# In[294]:


train['Sex'] = train['Sex'].map(lambda i : 1 if i=='male' else 0)
train.head()


# In[295]:


train['Embarked_S'] = train['Embarked'].map(lambda i: 1 if i=='S' else 0)
train['Embarked_C'] = train['Embarked'].map(lambda i: 1 if i=='C' else 0)
train['Embarked_Q'] = train['Embarked'].map(lambda i: 1 if i=='Q' else 0)
train.drop(['Embarked'],axis=1,inplace=True)
train.head()


# # Lets take a look at name feature
# 

# In[296]:


titles = [i.split(',')[1].split('.')[0].strip() for i in train['Name']]


# In[297]:


train['Title'] = pd.Series(titles)
train['Title'].head()


# In[298]:


train['Title'].value_counts()


# In[299]:


rare_surnames = ['Rev','Col','Mlle','Don','Mme','Jonkheer','the Countess']
mapping_other_surnames = {'Mr':1,
                         'Mrs':2,
                         'Miss':2,
                         'Master':1,
                         'Dr':3,
                         'Col':1,
                         'Major':3,
                         'Ms':2,
                         'Lady':2,
                         'Capt':3,
                         'Sir':1,
                         'Rare':4}
train['Title'] = train['Title'].replace(rare_surnames,'Rare')
train['Title'] = train['Title'].map(mapping_other_surnames)
train['Title']=train['Title'].astype(int)


# In[300]:


train['Title'].head()


# In[301]:


train.drop(['Name'],axis=1,inplace=True)
train.head()


# # Lets drop passenger Id and take a look at Cabin feature

# In[302]:


train.drop(['PassengerId'],axis=1,inplace=True)


# In[303]:


train.head()


# In[304]:


train['Cabin'].describe()


# In[305]:


train['Cabin'][1][0]


# In[306]:


train['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in train['Cabin']])
train['Cabin'].head()


# In[307]:


sns.countplot(train['Cabin'])


# In[308]:


sns.barplot(x='Cabin',y='Survived',data=train)


# In[309]:


train['Cabin'].value_counts()


# In[310]:


train = pd.get_dummies(train,columns=['Cabin'],prefix='Cabin')


# In[311]:


train.head()


# In[312]:


train.drop(['Ticket'],axis=1,inplace=True)


# In[313]:


train.head()


# In[314]:


train.shape


# # Doing the same for test dataset. So instead of writing line by line, we can write a function which has all the lines same as training Eda and feature engineering.

# In[315]:


test.head()


# In[316]:


test.isna().sum().sort_values(ascending=False)


# In[317]:


test[test['Fare'].isna()==True]


# In[318]:


test.shape


# In[319]:


# def __cleaner__(df):
test.drop(['PassengerId'],axis=1,inplace=True)


# In[320]:


titles = [i.split(',')[1].split('.')[0].strip() for i in test['Name']]
test['Title'] = pd.Series(titles)


# In[321]:


test['Title'].isna().sum()


# In[322]:


rare_surnames = ['Rev','Col','Mlle','Don','Mme','Jonkheer','the Countess']
mapping_other_surnames = {'Mr':1,
                         'Mrs':2,
                         'Miss':2,
                         'Master':1,
                         'Dr':3,
                         'Col':1,
                         'Major':3,
                         'Ms':2,
                         'Lady':2,
                         'Capt':3,
                         'Sir':1,
                         'Rare':4}
test['Title'] = test['Title'].replace(rare_surnames,'Rare')
test['Title'] = test['Title'].map(mapping_other_surnames)
test.head()


# In[323]:


test.drop(['Name'],axis=1,inplace=True)


# In[324]:


test.shape


# In[325]:


test['Sex'] = test['Sex'].map(lambda i : 1 if i=='male' else 0)

test['Embarked_S'] = test['Embarked'].map(lambda i: 1 if i=='S' else 0)
test['Embarked_C'] = test['Embarked'].map(lambda i: 1 if i=='C' else 0)
test['Embarked_Q'] = test['Embarked'].map(lambda i: 1 if i=='Q' else 0)
test.drop(['Embarked'],axis=1,inplace=True)

test.drop(['Ticket'],axis=1,inplace=True)

test['FamilySize'] = test['Parch'] + test['SibSp'] +1
test['Single'] = test['FamilySize'].map(lambda i: 1 if i==1 else 0)
test['Small'] = test['FamilySize'].map(lambda i: 1 if i==2 else 0)
test['Medium'] = test['FamilySize'].map(lambda i: 1 if 3<=i<=4 else 0)
test['Large'] = test['FamilySize'].map(lambda i: 1 if i>4 else 0)

test['Age'].fillna(test['Age'].median(),inplace=True)


# In[326]:


test['Fare'].fillna(test['Fare'].mean(),inplace=True)

test['Cabin'] = pd.Series([i[0] if not pd.isnull(i) else 'X' for i in test['Cabin']])
test = pd.get_dummies(test,columns=['Cabin'],prefix='Cabin')
print(test.shape)


# In[327]:


test.head()


# In[328]:


test.columns


# In[329]:


train.columns


# In[330]:


test.isna().sum().sort_values(ascending=False)


# In[331]:


train.isna().sum().sort_values(ascending=False)


# In[332]:


test[test['Title'].isna()==True]


# In[333]:


test['Title'].fillna('1',inplace=True)


# In[334]:


test.isna().sum().sort_values(ascending=False)


# In[335]:


test['Cabin_T']=0


# In[336]:


plt.figure(figsize=(20,12))
sns.heatmap(train.corr(),annot=True)


# In[337]:


plt.figure(figsize=(20,12))
sns.heatmap(test.corr(),annot=True)


# In[338]:


test.head()


# 
# ## Model building and performing cross validation

# In[339]:


len(train)


# In[340]:


X_train = train.drop(['Survived'],axis=1)
y_train = train['Survived']
print('Shape of X_train is : ',X_train.shape)
print('Shape of Y_train is :',y_train.shape)


# In[341]:


kfold = StratifiedKFold(n_splits=10)
classifiers = [
    SVC(),
    DecisionTreeClassifier(),
    AdaBoostClassifier(DecisionTreeClassifier(),learning_rate=0.1),
    RandomForestClassifier(n_estimators=50),
    GradientBoostingClassifier(),
    KNeighborsClassifier(),
    LogisticRegression(),
    LinearDiscriminantAnalysis(solver='eigen',shrinkage='auto'),
    MLPClassifier(learning_rate='adaptive')
]


# In[342]:


import warnings
warnings.filterwarnings('ignore')
results = []
for classifier in classifiers:
  results.append(cross_val_score(estimator=classifier,X=X_train,y=y_train,cv=kfold,scoring='accuracy'))


# In[343]:


results


# In[344]:


mean = []
std = []
for result in results:
  mean.append(result.mean())
  std.append(result.std())

result_df = pd.DataFrame({'Cross Validation Mean':mean,'Cross Validation Error':std,'Algorithms':['Suppor vector classifier',
                                                                                                  'Decision Tree classifier',
                                                                                                  'AdaBoosting classifier',
                                                                                                  'Random forest classifier',
                                                                                                  'Gradient boosting',
                                                                                                  'K Neighbours classifier',
                                                                                                  'Logistic Regression classifier',
                                                                                                  'Linear discriminant analysis',
                                                                                                  'Multi layer perceptron classifier']})
result_df


# In[345]:


sns.barplot(x='Cross Validation Mean',y='Algorithms',data=result_df)


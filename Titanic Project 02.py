#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[6]:


titanic=pd.read_csv('titanic_train.csv')


# In[7]:


titanic


# from the above cell we can see titanic dataset have 891 rows and 12 columns

# In[8]:


titanic.head()


# In[9]:


# checking for missing data
titanic.isnull().sum()


# so here we can see that column age have 177 , cabin have 687 and embarked have 2 missing values.

# # making column ('PassengerId') as index column

# In[10]:


titanic=titanic.set_index('PassengerId')
titanic.head()


# In[11]:


sns.heatmap(titanic.isnull(),cmap='viridis', yticklabels=False,cbar=False)
#yticklabels=False to hide all y axis numbers
#cbar=False means to hide the bar that comes at right hand side


# Roughly 20% of Age data is missing .The proportion of age is small enough for reasonable replacement from some form of imputation looking the Cabin columns we are missing to much data to do somethinguseful at basic level we drop it later or change it into 'cabin known: 1 or 0'

# In[12]:


titanic.dtypes


# Here we have int64(4), object(5), and float64(2) values in the dataset

# # EDA

# In[13]:


corr=titanic.corr()
corr


# In[14]:


plt.figure(figsize=(15,10))
sns.heatmap(corr,cmap='Blues',annot=True,linewidth=3)


# ->Here we can notice Parch(Number of Parents/Children Aboard) column is highly positive corelated with Survived column means there were high impact on survival
# 
# ->and Age is negatively corelated with survival that age has high negative impact on survival.

# In[15]:


sns.set_style("whitegrid")
sns.countplot(x="Survived", data=titanic)


# ->We can see here Data is not fully balance but we'll treat the imbalance.
# 
# ->We have around 330 survived and around 560 not survived people.

# In[16]:


sns.set_style("whitegrid")
sns.countplot(x="Survived", data=titanic,hue='Sex')


# Here we check count servived or and not survived column, there we see not survived male count is around 500 and not survived female count below than 100
# 
# and the other side we see in survived count females are more than males that means at the time of incident female were the prirority to be survived.

# In[17]:


sns.countplot(x='Pclass', data=titanic)


# we notice here most of the people were in Pclass 3, must be including staff and other workers on ship.

# In[19]:


sns.set_style("whitegrid")
sns.countplot(x="Survived",data=titanic,hue='Pclass',palette='rainbow')


# Here we in Pclass 3 having more not survived people that means most of the people were in Pclass 3.
# 
# There we notice first prirority were given to Pclass 1 get survive Pclass 3 was the last prirority.

# In[20]:


sns.distplot(titanic['Age'].dropna(),kde=False, color='darkred',bins=40)


# In[21]:


sns.countplot(x='SibSp',data=titanic)


# Here we see the people those are with their sibling.
# 
# around 600 people don't have siblings or spouse, arounf 200 people having 1 sibling or spouse and other people having more that 1 sibling and spouse.

# In[22]:


titanic['Fare'].hist(bins=40,color='green',figsize=(8,4))


# Here we see the fare of travel(people who bought ticket) most (around 375) of people we notice having 0 fare means they must be worked and staff member on ship

# In[23]:


sns.countplot(x='Parch', hue='Survived', data=titanic)


# # Data Cleaning

# In[24]:


titanic.describe()


# In[25]:


plt.figure(figsize=(10,5))
sns.boxplot(x='Pclass',y='Age',palette='winter', data=titanic)


# There are 3 passenger class in our dataset
# 
# We notice here wealtheir in higher class seems older which makes sense. we'll use this average age values to impute based on Pclass for age.
# 
# Here we can see Pclass 1 average of age is arond 37 and Pclass 2 it's around 29 Pclass 3 it's around 24
# 
# so filling the age NaN values with average of Age.

# In[26]:


# create a function to fill the missing age data
def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass==1:
            return 37
        
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age


# In[27]:


titanic['Age']=titanic[['Age','Pclass']].apply(impute_age,axis=1)


# In[28]:


sns.heatmap(titanic.isnull(),cmap='viridis',yticklabels=False, cbar=False)


# As we can see Age value has been filled.

# In[29]:


titanic.drop('Cabin', axis=1, inplace=True)


# We dropping the cabin column because it identity of cabins and most of values are NaN, won't help for prediction

# In[30]:


titanic.head()


# In[31]:


sns.heatmap(titanic.isnull(),cmap='viridis', yticklabels=False, cbar=False)


# In[32]:


titanic.loc[titanic['Embarked'].isnull()]


# So here from the above cell we can see in Embarked column have 2 null values.
# 
# so let's fill the NaN values with Mode of this column

# In[33]:


titanic['Embarked']=titanic['Embarked'].fillna(titanic['Embarked'].mode()[0])


# In[34]:


sns.heatmap(titanic.isnull(),cmap='viridis', yticklabels=False,cbar=False)


# Here we can notice null values in Embarked columns have been filled

# # Dropping unnecessary columns there will be no use of these columns for prediction.

# In[35]:


titanic.drop('Name', axis=1, inplace=True)


# In[36]:


titanic.drop('Ticket', axis=1, inplace=True)


# In[37]:


titanic.head()


# In[38]:


titanic.dtypes


# so here we have int64(4), object(2) and float64(2) values in the new data set
# 
# converting into categorical features
# 
# we need to convert categorical features into dummie variables using pandas otherwise our machine learning model won't be able to take those features as input

# In[39]:


titanic.isnull().sum()


# So here we have int64(4), object(2) and float64(2) values in the new data set
# 
# converting into categorical features
# 
# we need to convert categorical features into dummie variables using pandas otherwise our machine learning model won't be able to take those features as input

# In[40]:


titanic.isnull().sum()


# so here we can see we don't have any null value present in the dataset

# # Using LabelEncoder for converting categorical to numerical

# In[41]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
titanic['Sex']= le.fit_transform(titanic['Sex'])
titanic['Embarked']= le.fit_transform(titanic['Embarked'])


# In[42]:


titanic.head()


# In[43]:


titanic.isnull().sum()


# In[44]:


titanic.dtypes


# So here from the above 2 cells we can notice that now we don't have any null value present in the data set and data have int64(6) and float64(2) values

# # Outliers Removal

# In[46]:


plt.figure(figsize=(15,15))
for i in enumerate(titanic):
    plt.subplot(4,3,i[0]+1)
    sns.boxplot(titanic[i[1]])


# Here we can see few columns having Outliers present . So let's remove them.

# # zscore

# In[47]:


from scipy import stats
from scipy.stats import zscore
z= np.abs(zscore(titanic))
print(np.where(z>3))


# In[48]:


titanic_1= titanic[(z<3).all(axis=1)]
print("With outliers::", titanic.shape)
print("After removing outliers::",titanic_1.shape)


# using zscore method it removes 71 rows.

# # IQR method

# In[49]:


#IQR
from scipy import stats
IQR = stats.iqr(titanic[['Survived','Pclass','Parch','Age','Fare','SibSp','Parch']])
IQR


# In[50]:


Q1 = titanic.quantile(0.25)
Q3 = titanic.quantile(0.75)


# In[51]:


titanic_out = titanic[~((titanic < (Q1 - 1.5 * IQR)) |(titanic > (Q3 + 1.5 * IQR))).any(axis=1)]
print(titanic_out.shape)


# Using IQR method there is Huge data loss . So considering the ZSCORE Method

# In[52]:


titanic=titanic_1


# In[53]:


titanic.shape


# so here we have 820 rows and 8 columns after removing outliers.

# # Skewness Handling

# In[54]:


titanic.columns


# In[55]:


plt.figure(figsize=(25,20))
for i in enumerate(titanic):
    plt.subplot(3,4,i[0]+1)
    sns.distplot(titanic[i[1]],color='g')


# In[56]:


titanic.skew()


# We can notice that there is skewness in the columns

# We can notice that there is skewness in the columns
# 
# Pclass
# 
# Sex
# 
# SibSp
# 
# Parch
# 
# Fare
# 
# Embarked
# 
# but we only treat the numerical columns (given below) because other columns were changed catogorical to numerical.
# 
# Pclass
# 
# SibSp
# 
# Parch
# 
# Fare

# In[57]:


from sklearn.preprocessing import power_transform
titanic[['Pclass','SibSp','Parch','Fare']]=power_transform(titanic[['Pclass','SibSp','Parch','Fare']],method='yeo-johnson')


# In[58]:


titanic.skew()


# We can notice skewness almost removed from the targeted columns.

# # Dividing Data into x and y

# In[60]:


x=titanic.drop(['Survived'],axis=1)
y=titanic['Survived']


# In[61]:


x.shape


# In[62]:


y.shape


# # Scalling x values

# In[64]:


from sklearn.preprocessing import MinMaxScaler

sc=MinMaxScaler()
x=sc.fit_transform(x)


# In[65]:


pd.DataFrame(x)


# # Imbalanced Learn

# # Using RandomUnderSampler

# In[66]:


sns.set_style("whitegrid")
sns.countplot(x="Survived", data=titanic)


# We can see here Data is not fully balanced but we'll treat the imbalance
# 
# we have around 330 survived and 500 not survived people

# In[72]:


from imblearn.under_sampling import RandomUnderSampler


# In[ ]:





# In[ ]:





# In[70]:


rus = RandomUnderSampler(random_state=42)
x_rus, y_rus= rus.fit_resample(x,y)
print('Original Target dataset shape:',y.shape)
print('Resample Target dataset shape',y_rus.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# 

# In[ ]:





# In[ ]:





# # 

# In[ ]:





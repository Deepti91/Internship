#!/usr/bin/env python
# coding: utf-8

# In[97]:


import pandas as pd
import numpy as np
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


sns.set_style('darkgrid')
plt.rcParams['font.size'] = 15
plt.rcParams['figure.figsize'] = (10, 7)
plt.rcParams['figure.facecolor'] = '#FFE5b4'


# In[4]:


df = pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/happiness_score_dataset.csv')


# In[5]:



df


# In[7]:


df.head()


# In[8]:


df.tail()


# # Exploratory Data Analysis(EDA)

# In[9]:


#checking the dimension of the dataset
df.shape


# This dataset contains 158 rows and 12 columns, Out of which 1 is independent variable and other 10 are dependent variables

# In[10]:


df.columns


# In[11]:


df.columns.tolist()


# In[ ]:


#checking the types of columns


# In[12]:


df.dtypes


# There are 3 different types of data types object, float64 and int64 in this dataset.

# In[13]:


#checking for the null values
df.isnull().sum()


# In[14]:


df.info()


# As we can there is no null value present in this dataset

# In[15]:


#lets visualizing it using heatmap
sns.heatmap(df.isnull())


# And we can visualize that there is no missing data is present

# In[16]:


df.info()


# In[20]:


#Checking the value counts of each columns
for i in df.columns:
    print(df[i].value_counts())
    print("\n")


# In[28]:


#checking number of unique values in each column
df.nunique().to_frame("no of unique values")


# These are the unique values present in dataset.

# # Description of dataset

# In[29]:


#statistical summary of columns
df.describe()


# This gives the statistical information of the followiing information, This summary of the datasets looks perfect since there is no negative values/invalid values are present.
# From the description, we can observe the following.
# 
# 1. The counts of all the values are same that means there is no missing values.
# 2. The mean value is greater than Median(50%) in Happiness Score, Trust, Generosity, Dystopia Residual and Standard Error which means its skewed to Right Side.
# 3. The Median(50%) is greater than Mean in Happiness Rank, Family, Health and Freedom.
# 4. By summarizing, there is a huge difference between 75% and Max.

# # Data Vizualisation

# # Univariate Analysis

# In[34]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Happiness Rank', data = df)


# In[36]:


fig = plt.figure(figsize =(7,5))
sns.barplot(x = 'Happiness Score', y = 'Standard Error', data = df)


# In[51]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Family', data = df)


# In[52]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Health (Life Expectancy)', data = df)


# In[53]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Freedom', data = df)


# In[54]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Trust (Government Corruption)', data = df)


# In[55]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Generosity', data = df)


# In[56]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'Happiness Score', y = 'Dystopia Residual', data = df)


# In[61]:


print(df['Standard Error'].value_counts())
sns.countplot(x='Standard Error', data=df)
plt.show()


# In[62]:


print(df['Happiness Rank'].value_counts())
sns.countplot(x='Happiness Rank', data=df)
plt.show()


# In[63]:


print(df['Economy (GDP per Capita)'].value_counts())
sns.countplot(x='Economy (GDP per Capita)', data=df)
plt.show()


# In[64]:


print(df['Family'].value_counts())
sns.countplot(x='Family', data=df)
plt.show()


# In[65]:


print(df['Health (Life Expectancy)'].value_counts())
sns.countplot(x='Health (Life Expectancy)', data=df)
plt.show()


# In[66]:


print(df['Freedom'].value_counts())
sns.countplot(x='Freedom', data=df)
plt.show()


# In[67]:


print(df['Trust (Government Corruption)'].value_counts())
sns.countplot(x='Trust (Government Corruption)', data=df)
plt.show()


# In[68]:


print(df['Generosity'].value_counts())
sns.countplot(x='Generosity', data=df)
plt.show()


# In[70]:


print(df['Dystopia Residual'].value_counts())
sns.countplot(x='Dystopia Residual', data=df)
plt.show()


# In[82]:


df


# In[101]:


df.drop(['Country', 'Region', 'Happiness Rank'],axis=1, inplace=True)


# Here we are dropping columns that are not useful for the prediction model

# In[102]:


df.describe()


# In[87]:


sns.pairplot(df)


# # Checking Skewness

# In[103]:


df.head()


# In[104]:


df.columns


# In[110]:


columns=['Happiness Score', 'Standard Error', 'Economy (GDP per Capita)', 'Family', 'Trust (Government Corruption)', 'Generosity', 'Dystopia Residual']


# In[ ]:





# In[111]:


list(enumerate(columns))


# In[ ]:





# plt.figure(figsize=(12,19))
# for i in enumerate(columns):
#     plt.subplot(5,2,i[0]+1)
#     sns.distplot(df[i[1]])
# plt.show()
# df.skew()

# here we notice columns - Standard Error , Trust (Government Corruption) are highly skewed and few other columns- Family, Health(Life Expectancy) and Generosity are also little skewed

# # Check Outliers using zscore

# In[113]:


from scipy.stats import zscore


# In[114]:


plt.figure(figsize=(15,10))
for j in enumerate(columns):
    plt.subplot(3,3,j[0]+1)
    sns.boxplot(y=df[j[1]])


# In[115]:


zscore_outliers=np.abs(zscore(df))
threshold=3
print('Outliers:- \n',np.where(zscore_outliers>3))


# # Outliers Removal (Using zscore)

# In[116]:


dfzscore=df[(zscore_outliers<3).all(axis=1)]
print("Before removing outliers::",df.shape)
print("After removing outliers::",dfzscore.shape)


# # Using IQR

# In[118]:


from scipy import stats
IQR = stats.iqr(df)
IQR


# In[119]:


Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)


# In[120]:


IQR = df[~ ((df < (Q1 -1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
print(IQR.shape)


# Using zscore we can see there are 9 rows have been removed But by using IQR we can notice there is only 1 row has been removed . So here considering IQR method

# # Skewness removal

# In[121]:


df_corr=df.corr()
df_corr['Happiness Score']  # between 0.7 to 0.9 considered as high correlated


# As we can see columns- Economy (GSP per Capita), Family, Health(Life Expectancy) are highly corelated with the target data.

# In[122]:


df.skew()


# So here columns - Standard Error , Trust(Government Corruption) are highly skewed. And few other columns - Family , Health(Life Expectancy) , Generosity are also little skewed and highly correlated with target variable.
# 
# If we remove skewness from those columns who are highly correlated with target variable , that can affect the correlation of these columns .
# 
# So we only remove the skewness from those column, those are skewed and less correlated with target variable.

# In[123]:


# selecting column to remove skewness

skewed=['Standard Error', 'Trust (Government Corruption)', 'Generosity']
print('Before removing skewness:-')
plt.figure(figsize=(12,19))
for i in enumerate(skewed):
    plt.subplot(5,1,i[0]+1)
    sns.distplot(df[i[1]])
plt.show()


# In[124]:


# removing skewness:-

df[['Standard Error',
    'Trust (Government Corruption)',
    'Generosity']]=np.sqrt(df[['Standard Error','Trust (Government Corruption)', 'Generosity']])


# In[125]:


print('After removing skewness:-')
plt.figure(figsize=(12,19))
for i in enumerate(skewed):
    plt.subplot(5,2,i[0]+1)
    sns.distplot(df[i[1]])
plt.show()


# In[126]:


df.skew()


# so we can see that the skewness has been removed from the selected columns.

# # Dividing data into x and y

# In[129]:


x= df.drop(['Happiness Score'], axis=1)
y= df['Happiness Score']


# In[130]:


x.shape


# In[131]:


y.shape


# # Model Building

# In[134]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split


# # Spliting the data for training and testing

# In[135]:


# splitting the data into 76% training and 24% testing
x_train,x_test, y_train,y_test=train_test_split(x,y,test_size=0.24, random_state=43)


# In[136]:


x_train.shape


# In[137]:


x_train.shape


# In[138]:


x_test.shape


# In[139]:


y_test.shape


# here split the data for testing and training

# # Linear Regression

# In[141]:


LR=LinearRegression()
LR.fit(x_train,y_train)


# In[142]:


LR.coef_


# In[143]:


LR.score(x_train,y_train)


# In[144]:


LR_predict=LR.predict(x_test)


# In[145]:


print(LR_predict)


# In[146]:


print(y_test)


# In[147]:


#checking the model performance and accuracy using Mean Squared Error(MSE)
print(np.mean((LR_predict - y_test)**2))


# In[148]:


#checking the model performance and accuracy using mean Squared Error(MSE) and sklearn.metrics
print(mean_squared_error(LR_predict,y_test))


# In[150]:


#checking the model performance and accuracy using mean Squared Error(MSE) and sklearn.metrics
print(mean_squared_error(LR_predict,y_test))


# In[151]:


#checking the model performance and accuracy usin mean_absolute_error(MAE) and sklearn.matrics
print(mean_absolute_error(LR_predict,y_test))


# In[152]:


#checking the model performance and accuracy using r2_score and sklearn.metrics
print(r2_score(LR_predict,y_test))


# # SVR

# In[153]:


from sklearn.svm import SVR


# In[154]:


SVR(kernel= 'linear')


# In[155]:


svr_l=SVR(kernel='linear')
svr_l.fit(x_train,y_train)
print(svr_l.score(x_train,y_train))
svrpred_l=svr_l.predict(x_test)


# In[156]:


print('MSE:-',mean_squared_error(svrpred_l,y_test))
print('MAE:-',mean_absolute_error(svrpred_l,y_test))
print('r2_score:-',r2_score(svrpred_l,y_test))


# In[158]:


svr_p=SVR(kernel='poly')
svr_p.fit(x_train,y_train)
print(svr_p.score(x_train,y_train))
svrpred_p=svr_p.predict(x_test)


# In[159]:


print('MSE:-',mean_squared_error(svrpred_p,y_test))
print('MAE:-',mean_absolute_error(svrpred_p,y_test))
print('r2_score:-',r2_score(svrpred_p,y_test))


# In[160]:


svr_r=SVR(kernel='rbf')
svr_r.fit(x_train,y_train)
print(svr_r.score(x_train,y_train))
svrpred_r=svr_r.predict(x_test)


# In[161]:


print('MSE:-',mean_squared_error(svrpred_r,y_test))
print('MAE:-',mean_absolute_error(svrpred_r,y_test))
print('r2_score:-',r2_score(svrpred_r,y_test))


# # Saving the Model

# In[163]:


import joblib


# In[164]:


#save the best score model in joblib
joblib.dump(svr_p,'World_Happiness_Report_project')


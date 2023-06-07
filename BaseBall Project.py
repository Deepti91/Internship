#!/usr/bin/env python
# coding: utf-8

# In[7]:


# Importing requied library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# Importing the dataset

# In[8]:


df=pd.read_csv('baseball.csv')
df


# In[9]:


#Sample Dataset
df.sample()


# In[10]:


df.sample(5)


# In[11]:


df.min()


# In[12]:


df.max()


# In[13]:


df.mean()


# In[14]:


df.dtypes


# In[15]:


df.isnull().sum()


# In[16]:


sns.heatmap(df.isnull())


# In[17]:


#Column Names
column=df.columns
column


# In[18]:


#Informatio about the dataset
df.info()


# In[19]:


df.shape


# In[20]:


df['W'].unique()


# In[21]:


df['W'].nunique()


# While checking the Target variable for unique values we find there are 24 different types of values are present out of 30 entries which indicates this is a classification problem.

# In[22]:


#counting the unique values present in each column
for i in column:
    value_counts = df[i].value_counts()
    print('\n')
    print(value_counts)


# In[23]:


df.describe()


# Key Observations

# The columns looks good as tehre are no negetive or invalid values present and no null values as well. There are not much diffenence between the mean and 50% in most of the columns which means the column are slightly skewed. The standard deviation in most of the column indicates the dataset is widely spread except for the ERA, CG,SHO Colums. The difference between the 75% and max value is very high in R,H,HR, BB, SO,RA, ER columns indicating the presence of outliers.

# # Exploratory Data Analysis

# Univariate Analysis

# In[24]:


sns.histplot(df['W'])


# # Observation

# The target column 'W' (Win) is normally distributed. Highest Win (W) is in between 76-82 and Lowest Win (W) is in between 69-76. Overall Win (W) rate is between 63-100.

# In[25]:


plt.plot(df['R'])


# Observation:

# The column 'R' (Runs) is not normally distributed. Highest Runs (R) is in between 680-730 and No Runs (Runs) is in between 771-855. Overall Runs (R) rate is between 550-620.

# In[28]:


sns.histplot(df['AB'])


# Observation:

# The column 'AB' (At Bats) is normally distributed. Highest At Bats Rate (AB) is in between 5480-5520 and Lowest At Bats Rate (AB) is in between 5380-5430, 5530-5570 and 5610-5645. Overall At Bats Rate (AB) rate is between 5380-5645.

# In[29]:


sns.histplot(df['H'])


# Observation:

# The column 'H' (Hits) is not normally distributed.Highest Hits (H) is in between 1355-1385 and Lowest Hits (H) is in between 1420-1455. Overall Hits (H) rate is between 1322-1525.

# In[30]:


sns.histplot(df['2B'])


# Observation:

# The column '2B' (Doubles) is normally distributed.Highest Doubles (2B) is in between 272-284 and Lowest Doubles (2B) is in between 248-260. Overall Doubles (2B) rate is between 238-308.

# In[31]:


sns.histplot(df['3B'])


# Observation

# The target column '3B' (Triples) is normally distributed.Highest Triples (3B) is in between 25-32 and Lowest Triples (3B) is in between 13-25. Overall Triples (3B) rate is between 13-48.

# In[32]:


sns.histplot(df['HR'])


# Observation:

# The column 'HR' (Homeruns) is not normally distributed. Highest Homeruns (HR) is in between 168-184 and Lowest Homeruns (HR) is in between 188-210. Overall Homeruns (HR) rate is between 100-122.

# In[33]:


sns.histplot(df['BB'])


# Observation

# The column 'BB' (Walks) is normally distributed. Highest Walks (BB) is in between 470-505 and Lowest Walks (BB) is in between 505-540. Overall Walks (BB) rate is between 375-570.

# In[34]:


sns.histplot(df['SO'])


# Observation:

# The column 'SO' (Strikeouts) is normally distributed. Highest Strikeouts (SO) is in between 1250-1350 and Lowest Strikeouts (SO) is in between 500-1150 and 1440-1540. Overall Strikeouts (SO) rate is between 500-1540.

# In[37]:


sns.histplot(df['SB'])


# Observation:

# The column 'SB' (Stolen Bases) is normally distributed. Highest Stolen Bases (SB) is in between 58-90 and Lowest Stolen Bases (SB) is in between 105-118. Overall Stolen Bases (SB) rate is between 63-100.

# In[38]:


sns.histplot(df['RA'])


# Observation:

# The column 'RA' (Runs Allowed) is normally distributed. Highest Runs Allowed (RA) is in between 680-730 and Lowest Runs Allowed (RA) is in between 525-575. Overall Runs Allowed (RA) rate is between 525-840.

# In[39]:


sns.histplot(df['ER'])


# Observation

# The column 'ER' (Earned Runs) is normally distributed. Highest Earned Runs (ER) is in between 640-680 and Lowest Earned Runs (ER) is in between 425-525. Overall Earned Runs (ER) rate is between 525-800.

# In[41]:


sns.histplot(df['ERA'])


# Observation:

# The column 'ERA' (Earned Run Average ) is normally distributed. Highest Earned Run Average (ERA) is in between 3.8-4.2 and Lowest Earned Run Average (ERA) is in between 4.7-5.1 Overall Earned Run Average (ERA) rate is between 2.9-3.3'''

# In[42]:


sns.histplot(df['CG'])


# Observation

# The column 'CG' (Complete Games) is normally distributed. Highest Complete Games (CG) is in between 0-1.8 and No Complete Games (CGW) is in between 7.5-9. Overall Complete Games (CG) rate is between 0-11.

# In[43]:


sns.histplot(df['SHO'])


# Observation:

# The column 'SHO' (Shutouts) is normally distributed. Highest Shutouts (SHO) is in between 11.5-14.5 and Lowest Shutouts (SHO) is in between 16.5-18.5 Overall Shutouts (SHO) rate is between 4-21

# In[45]:


sns.histplot(df['SV'])


# Observation:

# The column 'SV' (Saves) is not normally distributed. Highest Saves (SV) is in between 33-39 and Lowest Saves (SV) is in between 28-34. Overall Saves (SV) rate is between 28-62.

# In[46]:


sns.histplot(df['E'])


# Observation:

# The column 'E' (Errors) is not normally distributed. Highest Errors (E) is in between 88-94 and Lowest Errors (E) is in between 101-114. Overall Errors (E) rate is between 75-82.

# In[48]:


plt.figure(figsize=(15,15))
plotnumber=1
for column in df:
    if plotnumber<=17:
        ax=plt.subplot(6,3,plotnumber)
        sns.distplot(df[column])
        plt.xlabel(column,fontsize =10)
    plotnumber+=1
plt.tight_layout()


# Observation:

# The target column 'W' and columns 'AB','3B','SB','CG', & 'ER' is normally distributed and having no skewness. Column 'R' is not normally distributed and it is skewed to the left and also Column 'H','HR','SV','E' are also not normally distributed but they are skewed to the right. Columns '2B','BB','SO','RA','ERA'and'SHO' looks normal but skewness is present.

# # Bivariate analysis

# In[49]:


sns.scatterplot(x='R',y='W',data=df)


# Observation

# We can see as Column Runs (R) is increasing Column Number of predicted wins (W) is also increasing, so Column 'R' is having positive correlation with Column 'W'. Data is not normally distributed , we can see in between 750-890 data is not linear.

# In[51]:


sns.scatterplot(x="AB",y="W",data=df)


# Observations

# We can see the data is normally distributed and Column 'AB' is negatively correlated with Column 'W'.

# In[54]:


sns.scatterplot(x="H",y="W",data=df)


# In[55]:


sns.catplot(x='2B',y='W',data=df,kind='bar')


# In[56]:


sns.scatterplot(x="3B",y="W",data=df)


# In[57]:


sns.scatterplot(x="HR",y="W",data=df)


# In[58]:


sns.scatterplot(x="SO",y="W",data=df)


# In[59]:


sns.scatterplot(x="SB",y="W",data=df)


# In[60]:


sns.scatterplot(x="RA",y="W",data=df)


# In[61]:


sns.scatterplot(x="ER",y="W",data=df)


# In[62]:


sns.scatterplot(x="ERA",y="W",data=df)


# In[63]:


sns.scatterplot(x="CG",y="W",data=df)


# In[64]:


sns.scatterplot(x="SHO",y="W",data=df)


# In[65]:


sns.scatterplot(x="SV",y="W",data=df)


# Multivariate analysis

# In[66]:


sns.pairplot(df,hue="W")


# In[67]:


#  Checking Correlation
df.corr()


# In[68]:


plt.figure(figsize=(20,15))
sns.heatmap(df.corr(),annot=True)


# Key Observations

# RA,ER,ERA are very negetively corrolating with target variable W. But ERA, ER, RA are positively correlating with each other.
# 
# R, 2B,HR, BB, SO, SHO AND SV ARE moderately correlating.
# 
# RA, ER,ERA also negetively correlating with SHO and SV.

# In[69]:


# Checking correlation by plotting barplot.
plt.figure(figsize=(12,7))
df.corr()['W'].sort_values(ascending=True).drop(['W']).plot(kind='bar',color='c')
plt.xlabel('Feature',fontsize=14)
plt.ylabel('Target',fontsize=14)
plt.title('Correlation',fontsize=18)
plt.show()


# Observation of the correlation:
# 
# Positively correlated with : 'H', 'CG', 'SO', 'HR', '2B', 'R', 'SHO', 'BB' and 'SV'
# 
# Negatively correlated with : 'ERA', 'RA', 'ER', '3B', 'SB', 'E' and 'AB'

# In[70]:


# Checking Outliers
coll=df.columns.values
ncol=17
nrows=30
plt.figure(figsize=(ncol,6*ncol))
for column in range(0,len(coll)):
    plt.subplot(nrows,ncol,column+1)
    sns.boxplot(data=df[coll[column]],color='pink',orient='v')
    plt.xlabel(column,fontsize = 15)
    plt.tight_layout()


# # Data Cleaning

# In[71]:


#removing the outliers by Z-Score method
from scipy.stats import zscore
z=np.abs(zscore(df))
z


# In[72]:


threshold=3
print(np.where(z>3))


# In[73]:


len(np.where(z>3)[0])


# In[74]:


df_new=df[(z<3).all(axis=1)]
df_new


# In[75]:


df.shape


# In[76]:


df_new.shape


# so after removing zscore 1 row is deleted as an outlier from 30 rows
# 
# Checking for skewness

# In[77]:


df_new.skew()


# Observation: Skewness threshold taken is +/-0.5. Columns which are having skewness: 'H','CG', 'SHO', 'SV' and 'E'. So, we will remove skewness from these columns.
# 
# The 'CG' column data is highly skewed

# # Removing Skewness

# In[78]:


#lets find the best method for skewness
from scipy import stats
from scipy.stats import skew, boxcox
def skee(a):
    model=[np.sqrt(a),np.log(a),stats.boxcox(a)[0]]
    print('original skewness is: ', a.skew())
    print('\n')
    for m in model:
        df_new=m
        print(skew(m))
        print('\n')


# In[79]:


skee(df_new['H'])


# In[80]:


df_new['H']=np.log(df_new['H'])


# In[81]:


skee(df_new['SHO'])


# In[82]:


df_new['SHO']=stats.boxcox(df_new['SHO'])[0]


# In[83]:


skee(df_new['SV'])


# In[84]:


df_new['SV']=stats.boxcox(df_new['SV'])[0]


# In[85]:


skee(df_new['E'])


# In[86]:


sns.distplot(df_new['H'])


# In[87]:


sns.distplot(df_new['SHO'])


# In[88]:


sns.distplot(df_new['SV'])


# In[89]:


sns.distplot(df_new['E'])


# Now the skewness in above columns looks good
# 
# So from the above observation we conclude that CG is not very well correlating with Target variable W, It also has outliers present and now we find it also has negetive values present in it so we will drop the column.

# In[90]:


df_new.drop(['CG'],axis=1,inplace=True)


# In[91]:


df_new.shape


# With this we are done with Data Analysis and Data Cleaning now we will move towards Data Preprocessing

# In[92]:


df_new.skew()


# Data preprocessing

# Separating features and labels

# In[93]:


# here we are sepearating our target variable from independent /feature variables
x=df_new.drop('W',axis=1)
y=df_new['W']


# In[94]:


x


# In[95]:


x.shape


# In[96]:


y


# In[97]:


y.shape


# Dealing with class imbalance

# In[99]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x = pd.DataFrame(scaler.fit_transform(x), columns = x.columns)
x


# In[100]:


x.shape


# In[101]:


x


# Here in fit method we are giving the data onthe method and the method will do diffenent analysis, it will find out relationship pattern and then we are using transform methos which will on the basis of previous learning the data will be changed
# 
# We have scaled the data using Standard Scalarization method to overcome the issue of biasness

# Checking for Multicolinearity

# In[102]:


from statsmodels.stats.outliers_influence import variance_inflation_factor
vif=pd.DataFrame()
vif['VIF Values']=[variance_inflation_factor(x.values,i) for i in range (len(x.columns))]
vif['Features']=x.columns
vif


# The VIF value is more than 10 in AB, RA, ER and ERA Column. But since ER is having the highest variance we will drop that column

# In[103]:


x.drop('ER',axis = 1, inplace=True)


# In[104]:


x


# In[105]:


vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
vif['Features']=x.columns
vif


# After dropping ER column we can see significant changes in AB RA and ERA columns but we will still try dropping RA and check again.

# In[106]:


x.drop('RA',axis=1, inplace=True)


# In[107]:


vif=pd.DataFrame()
vif['VIF values']=[variance_inflation_factor(x.values,i) for i in range(len(x.columns))]
vif['Features']=x.columns
vif


# So Multicolinerity is removed from the columns And finally after dropping RA column all the columns has vif value under 10.

# # Building the model

# In[108]:


from sklearn.model_selection import train_test_split, GridSearchCV,cross_val_score
from sklearn.linear_model import LinearRegression, SGDRegressor,Ridge, Lasso
from sklearn.metrics import classification_report,confusion_matrix,roc_curve,accuracy_score
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


# In[109]:


maxAccu=0
maxRS=0
for i in range(1,200):
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.20,random_state=i)
    lr=LinearRegression()
    lr.fit(x_train,y_train)
    pred=lr.predict(x_test)
    acc=r2_score(y_test,pred)
    if acc>maxAccu:
        maxAccu=acc
        maxRS=i
print("Best accuracy is ",maxAccu,"at random state",maxRS)


# The best accuracy score is 96% at random state 195

# In[110]:


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor


# In[111]:


lr=LinearRegression()
RFR=RandomForestRegressor()
knn=KNeighborsRegressor()
gbr=GradientBoostingRegressor()
sv=SVR()
dtc=DecisionTreeRegressor()


# In[112]:


x_train.shape,y_train.shape, x_test.shape,y_test.shape


# In[113]:


print(lr)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
lr.fit(x_train,y_train)
pred=lr.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# In[114]:


print(RFR)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
RFR.fit(x_train,y_train)
pred=RFR.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# In[115]:


print(gbr)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
gbr.fit(x_train,y_train)
pred=gbr.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# In[116]:


print(knn)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
knn.fit(x_train,y_train)
pred=knn.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# In[117]:


print(sv)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
sv.fit(x_train,y_train)
pred=sv.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# In[118]:


print(dtc)
print('\n')
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=195)
dtc.fit(x_train,y_train)
pred=dtc.predict(x_test)
print('Mean Squared Error:', mean_squared_error(y_test,pred))
r2 = r2_score(y_test,pred)
print('R-squared score:',r2)
print('\n')


# # CROSS VALIDATION SCORE

# In[119]:


models=[lr,RFR,gbr,knn,sv,dtc]
for m in models:
    
    score=cross_val_score(m,x,y,cv=4)
    print(m,'score is:')
    print(round((score.mean()),3))
    print('\n')


# From the above result we find that the r2_score and cv score results are negetive in case of support vector regressor and DecisionTreeRegressor. This suggests that the models are not fitting the data well and may not generalize well to unseen data. So I will choose Linear Regressor as the best model because it has got least difference between r2 score and cross validation score.

# # Hyper parameter tuning

# In[120]:


# Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Create the Linear Regression model
lr=LinearRegression()

# Define the hyperparameters to tune
parameters = {'fit_intercept': [True, False], 'normalize': [True, False]}

# Create the GridSearchCV object
grid_search = GridSearchCV(estimator=lr, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)

# Fit the model to the training data
grid_search.fit(x_train, y_train)

# Get the best parameters and best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print('Best Parameters:', best_params)
print('Best Score (Negative MSE):', best_score)

# Use the best model to make predictions on the test data
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
print('Mean Squared Error:', mse)

# Calculate R-squared score
r2 = r2_score(y_test, y_pred)
print('R-squared score:', r2)


# In[121]:


#Visualize the result
plt.scatter(y_test,y_pred)
plt.xlabel('True Values')
plt.ylabel('Predicted Values')
plt.title('True vs Predicted Values')
plt.show()


# In[122]:


# Interpret the model coefficients
coefficients = best_model.coef_
intercept = best_model.intercept_
print('Intercept:', intercept)
print('Coefficients:', coefficients)


# In[123]:


# Predictions on new data
new_data = np.array(y_test)
new_predictions = np.array(best_model.predict(x_test))
Baseball_Case_Study = pd.DataFrame({"Original":new_data,"Predicted":new_predictions}, index=range(len(new_data)))
print(Baseball_Case_Study)


# # Saving the model

# In[124]:


import joblib
joblib.dump(lr,'baseball_pred.obj')


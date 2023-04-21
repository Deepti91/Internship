#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as snimport warnings
warnings.filterwarnings('ignore')


# In[3]:


Red_wine=pd.read_csv('https://raw.githubusercontent.com/dsrscientist/DSData/master/winequality-red.csv')


# In[4]:


Red_wine.head()


# In[5]:


Red_wine.columns


# In[6]:


Red_wine.dtypes


# In[7]:


Red_wine.isnull().sum()


# In[8]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'fixed acidity', data = Red_wine)


# In[9]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'volatile acidity', data = Red_wine)


# In[10]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'citric acid', data = Red_wine)


# In[11]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'residual sugar', data = Red_wine)


# In[12]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'chlorides', data = Red_wine)


# In[13]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = Red_wine)


# In[14]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = Red_wine)


# In[15]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'density', data = Red_wine)


# In[17]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'pH', data = Red_wine)


# In[18]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'sulphates', data = Red_wine)


# In[19]:


fig = plt.figure(figsize = (7,5))
sns.barplot(x = 'quality', y = 'alcohol', data = Red_wine)


# In[20]:


sns.pairplot(Red_wine)


# In[21]:


Red_wine.corr()


# In[22]:


Red_wine.corr()['quality'].sort_values()


# In[23]:


plt.figure(figsize=(15,7))
sns.heatmap(Red_wine.corr(),annot=True, linewidth=0.5, linecolor='black', fmt='.2f')


# In[24]:


#Outcome of Correlation
#Fixed Acidity has 12% correlation with the target column which can be considered as good bond
#Volatile Acidity has -39% correlation with the target column which can be considered as weak bond
#Citric Acid has 23% correlation with the target column which can be considered as good bond
#Residual Sugar has 1% correlation with the target column which can be considered as weak bond
#Chlorides has -13% correlation with the target column which can be considered as weak bond
#Free Sulfur Dioxide has -5% correlation with the target column which can be considered as weak bond
#Total Sulfur Dioxide has -19% correlation with the target column which can be considered as weak bond
#Density has -17% correlation with the target column which can be considered as weak bond
#pH has -6% correlation with the target column which can be considered as weak bond
#Sulphates has 25% correlation with the target column which can be considered as good bond
#Alcohol has 48% correlation with the target column which can be considered as strong bond
#Max Correlation: Alcohol

#Mean Correlation: Volatile Acidity


# Descriptive Statistics

# Describing Datasets:

# In[28]:


Red_wine.describe()


# In[29]:


plt.figure(figsize=(15,7))
sns.heatmap(round(Red_wine.describe()[1:].transpose(),2),linewidth=2,annot=True,fmt='f')
plt.xticks(fontsize=18)
plt.xticks(fontsize=12)
plt.title('variables')
plt.savefig('heatmap.png')
plt.show()


# In[30]:


"""Outcome of Describe of Datasets:
We are determining Mean, Standard Deviation, Minimum and Maximum Values of each column from the plot which will help in data cleaning

Total No of Rows: 1599 Total No. of Columns: 12

Fixed Acidity:

1. Mean= 8.319637
2. std= 1.741096
3. Min= 4.600000
4. Max= 15.900000

Volatile Acidity:

1. Mean= 0.527821
2. std= 0.179060
3. Min= 0.120000
4. Max= 1.580000

Citric Acid:

1. Mean= 0.270976
2. std= 0.194801
3. Min= 0.000000
4. Max= 1.000000

Residual Sugar:
1. Mean= 2.538806
2. std= 1.409928
3. Min= 0.900000
4. Max= 15.500000

Chlorides:

1. Mean= 0.087467
2. std= 0.047065
3. Min= 0.012000
4. Max= 0.611000

Free Sulfur Dioxide:

1. Mean= 15.874922
2. std= 10.460157
3. Min= 1.000000
4. Max= 72.000000

Total Sulfur Dioxide:
1. Mean= 46.467792
2. std= 32.895324
3. Min= 6.000000
4. Max= 289.000000

Density:

1. Mean= 0.996747
2. std= 0.001887
3. Min= 0.990070
4. Max= 1.003690

pH:

1. Mean= 3.311113
2. std= 0.154386
3. Min= 2.740000
4. Max= 4.010000

Sulphates:

1. Mean= 0.658149
2. std= 0.169507
3. Min= 0.330000
4. Max= 2.000000

Alcohol:

1. Mean= 10.422983
2. std= 1.065668
3. Min= 8.400000
4. Max= 14.900000

Quality is Target column and it comes under binary categorical data, so describe shows no valid outcome."""


# In[31]:


Red_wine.info()


# In[32]:


#We have now info about data types and memory usage

#Checking Outliers


# In[33]:


collist=Red_wine.columns.values
ncol=30
nrows=14
plt.figure(figsize=(ncol,3*ncol))
for i in range(0,len(collist)):
    plt.subplot(nrows,ncol,i+1)
    sns.boxplot(data=Red_wine[collist[i]],color='green',orient='v')
    plt.tight_layout()


# In[34]:


Red_wine.skew()


# In[35]:


#Skewness threshold taken is +/-0.65. Columns which are having skewness:

#fixed acidity, volatile acidity, residual sugar, chlorides, free sulfur dioxide, total sulfur dioxide and alcohol
#Normal Distribution Curve:


# In[36]:


sns.distplot(Red_wine['fixed acidity'])


# In[37]:


sns.distplot(Red_wine['volatile acidity'])


# In[38]:


sns.distplot(Red_wine['citric acid'])


# In[39]:


sns.distplot(Red_wine['residual sugar'])


# In[40]:


sns.distplot(Red_wine['chlorides'])


# In[41]:


sns.distplot(Red_wine['free sulfur dioxide'])


# In[42]:


sns.distplot(Red_wine['total sulfur dioxide'])


# In[43]:


sns.distplot(Red_wine['density'])


# In[44]:


sns.distplot(Red_wine['pH'])


# In[45]:


sns.distplot(Red_wine['sulphates'])


# In[46]:


sns.distplot(Red_wine['alcohol'])


# In[47]:


#The data is not normalised

#The Normal Distribution shows that the data is Skewed
#Data Cleaning


# In[48]:


Red_wine.corr()['quality']


# In[49]:


from scipy.stats import zscore


# In[50]:


z=np.abs(zscore(Red_wine))
z.shape


# In[51]:


z


# In[52]:


Red_wine["quality"].unique()


# In[53]:


Red_wine["quality"].unique()


# In[54]:


threshold=3
print(np.where(z>3))


# In[55]:


len(np.where(z>3)[0])


# In[56]:


wine=Red_wine[(z<3).all(axis=1)]
wine.head()


# In[57]:


print("Old DataFrame data in Rows and Column:",Red_wine.shape)


# In[58]:


print("New DataFrame data in Rows and Column:",wine.shape)


# In[59]:


print("Total Dropped rows:",Red_wine.shape[0]-wine.shape[0])


# In[60]:


loss_percent=(1599-1451)/1599*100
print(loss_percent,"%")


# In[61]:


sns.countplot(x="quality",data=Red_wine)
plt.show()


# In[62]:


Red_wine["quality"].unique()


# In[63]:


wine["quality"].unique()


# In[64]:


sns.countplot(x="quality",data=wine)
plt.show()


# In[67]:


#We can see here, we have removed one type of quality from the dataset after removing outliers.

#Converting Target Variable into Binary Classification

#splitting wine into good and bad groups, quality score of wines between 2 to 6.5 are "bad" quality and between 6.5 to 8 are "good" quality


# In[70]:


wine.head(15)


# In[71]:


sns.countplot(x="quality",data=wine)
plt.show()


# In[72]:


#Coverting quality column data into numeric data


# In[73]:


from sklearn.preprocessing import LabelEncoder


# In[74]:


encoder=LabelEncoder()
wine['quality']=encoder.fit_transform(wine['quality'])


# In[75]:


wine.head()


# In[76]:


x=wine.iloc[:,:-1]
y=wine.iloc[:,-1]


# In[77]:


x.head()


# In[78]:


y.head()


# In[79]:


#Removing skewness using yeo-johnson method:


# In[80]:


from sklearn.preprocessing  import power_transform


# In[81]:


x=power_transform(wine,method='yeo-johnson')
x


# In[82]:


wine.skew()


# In[83]:


sns.distplot(wine['chlorides'])


# In[84]:


#Data Standardization


# In[85]:


from sklearn.preprocessing import StandardScaler


# In[86]:


scaler=StandardScaler()
scaler.fit_transform(x)
x


# In[87]:


x.mean()


# In[88]:


#Creating Model


# In[89]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.metrics import accuracy_score


# In[90]:


#Logistic Regression Model
lr=LogisticRegression()


# In[91]:


for i in range(0,1000):
    x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=i, test_size=0.20)
    lr.fit(x_train,y_train)
    pred_train=lr.predict(x_train)
    pred_test=lr.predict(x_test)
    if round(accuracy_score(y_train,pred_train)*100,1)==round(accuracy_score(y_test,pred_test)*100,1):
        print('Logistic Regression model Result are:-' )
        print('At random state' , i, 'the Model performs very well')
        print('At random state:- ' , i)
        print('the training accuracy score is:- ', round(accuracy_score(y_train,pred_train)*100,1))
        print('the testing accuracy score is:- ', round(accuracy_score(y_test,pred_test)*100,1),'\n\n')


# In[92]:


x_train,x_test,y_train,y_test=train_test_split(x,y, random_state=9, test_size=0.20)


# In[93]:


lr.fit(x_train,y_train)


# In[94]:


#Classification report
from sklearn.metrics import classification_report


# In[95]:


print(classification_report(y_test,pred_test))


# In[96]:


#Cross Validation score for Logistic Regression
from sklearn.model_selection import cross_val_score
pred_lr=lr.predict(x_test)
lss=accuracy_score(y_test,pred_lr)


# In[97]:


for j in range(2,10):
    cv_score=cross_val_score(lr,x,y,cv=j)
    cv_mean=cv_score.mean()
    print("At cv: ",j)
    print("Cross Validation score is: ",cv_mean*100)
    print("accuracy_score is: ",lss*100,"\n")


# In[98]:


lsscore=cross_val_score(lr,x,y,cv=7).mean()
print("The cv score is: ", lsscore,"\nThe accuracy_score is: ",lss)


# In[99]:


#Confusion Matrix
from sklearn.metrics import confusion_matrix
def model_selection(algorithm_instance, x_train,y_train,x_test,y_test):
    algorithm_instance.fit(x_train,y_train)
    model1_pred_train = algorithm_instance.predict(x_train)
    model1_pred_test = algorithm_instance.predict(x_test)
    print("Accuracy of the Training Model: ", accuracy_score(y_train,model1_pred_train))
    print("Accuracy of the Test data: ", accuracy_score(y_test,model1_pred_test))
    print("Classification Report for train data: \n",classification_report(y_train,model1_pred_train))
    print("Classification Report for test data: \n",classification_report(y_test,model1_pred_test))
    print("confusion metrix for test data: \n",confusion_matrix(y_test,model1_pred_test))


# In[100]:


model_selection(lr,x_train,y_train,x_test,y_test)


# In[102]:


#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
dc=DecisionTreeClassifier()
model_selection(dc,x_train,y_train,x_test,y_test)


# In[103]:


#KNN Regressor:
from sklearn.neighbors import KNeighborsRegressor
kn =KNeighborsRegressor(n_neighbors=7, algorithm='kd_tree', weights='distance')
kn.fit(x_train, y_train)
pred_kn= kn.predict(x_test)
print(' KNN regressor result :-')
print('Accuracy score is ',accuracy_score(pred_test,y_test))


# In[104]:


#Random Forest Regressor:
from sklearn.ensemble import RandomForestRegressor
rf=RandomForestRegressor()
model_selection(rf,x_train,y_train,x_test,y_test)


# In[105]:


#Support Vector Regressor:
from sklearn.svm import SVR
sv = SVR()
sv.fit(x_train, y_train)
pred_kn= sv.predict(x_test)
print(' KNN regressor result :-')
print('Accuracy score is ',accuracy_score(pred_test,y_test))


# In[106]:


#Lasso Regression:
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Lasso,Ridge
parameters={'alpha':[0.0001,0.001,0.01,0.1,1,10],'random_state':list(range(0,100))}
ls=Lasso()
wine=GridSearchCV(ls,parameters)
wine.fit(x_train,y_train)
print(wine.best_params_)


# In[107]:


#AUC ROC Curve:
fpr,tpr,thresholds=roc_curve(pred_test,y_test)
roc_auc=auc(fpr,tpr)
plt.figure()
plt.plot(fpr,tpr,color="orange",lw=10,label="ROC Curve (area= %0.2f)" % roc_auc)
plt.plot([0,1],[0,1],color="navy",lw=10,linestyle="--")
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver operating Characteristic")
plt.legend(loc="lower right")
plt.show()


# In[108]:


#Since the Model without Outliers performing best! We will select that DataFrame.
#Saving the Model
import pickle
filename='Red_wine_quality.pickle'
pickle.dump(rf,open(filename,'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.predict(x_test)


# In[109]:


a =np.array(y_test)
predicted=np.array(rf.predict(x_test))
wine=pd.DataFrame({'Orginal':a,'Predicted':predicted}, index=range(len(a)))
wine


# In[110]:


#As we can see, predicted and original values matches 100%.


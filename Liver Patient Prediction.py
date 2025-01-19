#!/usr/bin/env python
# coding: utf-8

# # Project:- "Liver patient diseases dataset to predict liver diseases"

# ## Team ID : PTID-CDS-JUL-23-1658

# # Task 1:-Prepare a complete data analysis report on the given data.

# ### Domain analysis
# * Patients with Liver disease have been continuously increasing because of excessive consumption of alcohol, inhale of harmful gases, intake of contaminated food, pickles and drugs.
# * Excessive consumption of alcohol leads to Cirrhosis based liver disease.When consumption of excessive alcohol then start first stage is "Alcoholic fatty liver disease" in this stage start fat accumulate around liver. second stage "Acute alcoholic hepatitis" in this stage swelling liver and third stage is "Alcoholic cirrhosis" in this stage Cirrhosis can lead to liver failure.
# * Inhale harmful gases to lead liver disease because of many chemicals that are intentionally or unintentionally inhaled or consumed can have toxic effects on the liver.Since the liver is continually targeted by chemicals, extended chemical exposure can cause genetic mutations that cause cancer.
# * COntaminated food or sugar,white bread, red meat that increase fat around liver and that's lead to cirrhosis disease of liver and damage livers.
# * Large amount of drugs leads to liver damage like medicine example painkillers and fever reducers that contain acetaminophen are a common cause of liver injury, particularly when taken in doses greater than those recommanded by doctors.

# ## Dataset

# * This data set contains 416 liver patient records and 167 non liver patient records collected from North East of Andhra Pradesh, India. The "Target" column is a class label used to divide groups into liver patients (liver disease) or not (no disease). This data set contains 441 male patient records and 142 female patient records.
# 
# 

# * As per dataset statement classified two group like liver patients and not liver patients. so it is classification problem.

# # Importing Required Libraries

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy.stats import stats
import sweetviz as sv
import joblib
import xgboost
from xgboost import XGBClassifier,XGBRFClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV,KFold,RandomizedSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,f1_score,recall_score,precision_score,roc_curve,roc_auc_score
from collections import Counter
import warnings
warnings.filterwarnings("ignore")


# ## Step 2:-  Data collection

# In[2]:


# Data successfully loading.
data = pd.read_csv("Indian_Liver_Patient_Dataset.csv")
data


# In[3]:


columns_name=["Age","Gender","TL_BL","DRT_BL","ALP","ALT","AST","T_Protiens","Albumin","A_G","Target"]
data1=pd.read_csv("Indian_Liver_Patient_Dataset.csv",names=columns_name)


# In[4]:


data1.head(50)


# In[5]:


bold="\033[1m"
reset="\033[0m"
print(bold+"Top 10 datasets:-\n\n"+reset,data1.head(10))    # Top 10 datsets.
print('\n\n')
print(bold+"\nLast 10 datsets:-\n\n"+reset,data1.tail(10))    # last 10 datasets


# In[6]:


print(bold+"Shape of datasets:-"+reset,data.shape)


# * In our datasets total 11 features and 582 records

# ## Step 3 :- Identify dependent and independent variables.

# ### Dependent Variables
# * As per dataset statement target is find out liver disease or not like 
#        * 1 : patient with  liver disease
#        * 2: patient with no liver disease
#        
# ### Independent Variables
# * Rest of all features are independent variables.
# 

# # Step 4:- Exploratory Data Analysis(EDA)

# ### Task 1:-Prepare a complete data analysis report on the given data.

# In[7]:


# Check how many dublicate value in dataset
Duplicate=data1.duplicated().sum()
print("Total_duplicate value is:",Duplicate)


# In[8]:


data1.duplicated().value_counts()


# In[9]:


data1.loc[data1.duplicated()]

data1.drop_duplicates(inplace=True)# Verify dublicate data entry droped.
data1.duplicated().value_counts()
# * For this duplicated value not have any impact on our data analysis model.

# In[10]:


# Checking features information.
data1.info()


# * Gender is Object dataset.
# * Int value is a 5.
# * float value is a 5.

# In[11]:


# Check missing value in dataset.
data1.isna().sum()


# * In above datasets find out information that one features have missing value.
#        * A/G is 4 missing value.

# In[12]:


# Checking statistical information.
data1.describe()


# In[13]:


# Checking categorical data
data1.describe(include=["O"]).T


# In[14]:


data1['Target'].dtypes


# #### As per statistical information collect some points.
# * Age between 4 years to 90 years.
# * Total_Bilirubin having major outliers.
# * Direct_Bilirubin having minor outliers.
# * Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase having major outliers.
# * Total_Protiens,Albumin,Albumin_Globulin_Ratio are generalize dataset form.

# In[15]:


plt.figure(figsize=(10,10))
ah=sns.countplot()
plt.xlabel("Target",fontsize=20)
plt.ylabel("count",fontsize=20)
plt.title("Liver disease total count",fontsize=30)
plt.show()


# In[16]:


# Map the data are 1 indicate 'Liver disease' and 0 indicate 'Not liver disease'.
data1["Target"]=data1["Target"].map({1:1,2:0})


# In[17]:


plt.figure(figsize=(10,10))
ax=sns.countplot(data=data1,x="Gender",hue="Target")
plt.xlabel("Gender",fontsize=20)
plt.ylabel("Count records",fontsize=20)
plt.title("Gender vs Disease",fontsize=30)
plt.legend(['Non liver disease','Liver disease'])
#plt.grid()
plt.show()


# * 324 male candidate have liverdisease and 117 female candidate have not liver disease.
# * Data is imbalanced because 142 female patient and 441 male patient same as 416 patient liver disease 167 patient not liver patient disease.

# In[18]:


plt.figure(figsize=(20,20))
ad=sns.countplot(data=data1,x="Age",hue="Target")
plt.xlabel("Age",fontsize=20)
plt.ylabel("Count of records",fontsize=20)
plt.title("Age vs Disease graph",fontsize=20)
plt.legend(['Non liver disease','Liver disease'])
plt.show()


# ### On visual representation find out
# * 60 years old have highest liver disease.
# * 45 years old have second highest liver disease.

# In[19]:


# Checking distribution on bases of gender datasets.
plt.figure(figsize=(20,15))
plotnumber=1
for i in data1:
    if plotnumber<12:
        plot=plt.subplot(6,2,plotnumber)
        sns.histplot(data1,x=data1[i],kde=True)
        plt.xlabel("Target",fontsize=20)
        plt.ylabel(i,fontsize=15)
    plotnumber+=1
plt.tight_layout()


# * Normaly distribution data
#      * Albumin_Globulin vs Target 
#      * Age vs Target
# * Not normaly distribution data 
#      * Albumin vs Target- Uniform distribution 
#      * Total_protiens vs Target- Uniform distribution
#      * Alkaline_Phosphotase vs Target- Positive skewed         
#      * Alamine_Aminotransferase vs Target -Positive skewed
#      * Total_Bilirubin  vs Target- Positive skewed         
#      * Direct_Bilirubin vs Target- Positive skewed
#      * Having outliers in data.
# 

# ## Step 5:- Data Preprocessing

# ## Fix missing value in our datasets

# In[20]:


# Check proper location of missing value
data1.loc[data1["A_G"].isna()]


# In[21]:


# Fix the missing value in data.
data1["A_G"].fillna(data1["A_G"].mean(),inplace=True)


# In[22]:


# Not available in any missing value.
data1.A_G.isnull().sum()


# In[23]:


data1.loc[data1["A_G"].isna()]


# ### Data Encoding

# In[24]:


# 0 use for female and 1 use for male.
le=LabelEncoder()
data1["Gender"]=le.fit_transform(data1["Gender"])


# In[25]:


# Gender value convert into Numerical form.
data1


# ### Remove Outliers in our dataset.

# ### Total_Bilirubin

# In[26]:


data1.TL_BL.unique()


# In[27]:


data1.TL_BL.value_counts()


# In[28]:


sns.set(style="whitegrid")
plt.figure(figsize=(20,9))
sns.boxplot(x="TL_BL",
            color='r',
            showmeans=True,       # Display mean value in graph.
            meanprops={'marker':"o",'markeredgecolor':"black"}, # Mean value display in round bullet and black edge marking
            data=data1)
plt.show()


# In[29]:


# Checking lower limit and upper limit.
IQR= stats.iqr(data1.TL_BL,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.TL_BL.quantile(0.25)
print("Q1:",Q1)

Q3=data1.TL_BL.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+ 0.3*IQR
print("Upper_limit:",Upper_limit)


# In[30]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="TL_BL",kde=True,hue="Target")
plt.grid()
plt.xlabel("TL_BL",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.plot()


# * Total_bilirubin's are not normal distribution and right side skewed.

# In[31]:


x=data1.loc[data1["TL_BL"]<Lower_limit]
x


# In[32]:


y=data1.loc[data1["TL_BL"]>Upper_limit]
y


# In[33]:


# outliers present in percentage.
(len(x+y)/len(data1))*100


# In[34]:


data1.loc[data1["TL_BL"]>Upper_limit,"TL_BL"]=data1["TL_BL"].mean()


# In[35]:


# Remove outliers.
data1.loc[data1["TL_BL"]>Upper_limit]


# ### Direct_Bilirubin

# In[36]:


data1.DRT_BL.unique()


# In[37]:


data1.DRT_BL.value_counts()


# In[38]:


plt.figure(figsize=(20,9))
sns.boxplot(x="DRT_BL",
           showmeans=True,
           color='r',
           data=data1)


# In[39]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="DRT_BL",hue="Target",kde=True)
plt.grid()
plt.xlabel("DRT_BL",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# * Direct_Bilirubin's graph is not normaly distribution and skewed in right side.

# In[40]:


IQR=stats.iqr(data1.DRT_BL,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.DRT_BL.quantile(0.25)
print("Q1:",Q1)

Q3=data1.DRT_BL.quantile(0.75)
print("Q3",Q3)

Lower_limit=Q1-0.1*IQR
print("Lower_limit:",Lower_limit)

Upper_limit=Q3+2*IQR
print("Upper_limit",Upper_limit)


# In[41]:


k=data1.loc[data1["DRT_BL"]<Lower_limit]
k


# In[42]:


l=data1.loc[data1["DRT_BL"]>Upper_limit]
l


# In[43]:


data1.loc[data1["DRT_BL"]>Upper_limit,"DRT_BL"]=data1["DRT_BL"].mean()


# In[44]:


# Upper limit of outliers removed with fill mean value.
data1.loc[data1["DRT_BL"]>Upper_limit]


# ### Alkaline_Phosphotase

# In[45]:


data1.ALP.unique()


# In[46]:


data1.ALP.value_counts()


# In[47]:


plt.figure(figsize=(20,9))
sns.boxplot(x="ALP",
           showmeans=True,
           color='r',
           data=data1)


# In[48]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="ALP",hue="Target",kde=True)
plt.grid()
plt.xlabel("ALP",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# * Identify as per graph laying on right side skewed and not normally distributed.

# In[49]:


IQR=stats.iqr(data1.ALP,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.ALP.quantile(0.25)
print("Q1:",Q1)

Q3=data1.ALP.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-1*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+10*IQR
print("Upper limit:",Upper_limit)


# In[50]:


m=data1.loc[data1["ALP"]<Lower_limit]
m


# In[51]:


n=data1.loc[data1["ALP"]>Upper_limit]
n


# In[52]:


# Total outliers available in Alkaline_Phosphotase features
(len(m+n)/len(data1))*100


# In[53]:


data1.drop(data1[data1["ALP"]>Upper_limit].index,inplace=True)

data1.loc[data1["ALP"]>Upper_limit,"ALP"]=data1["ALP"].median()
# In[54]:


# Remove all outliers as a mean value.
data1.loc[data1["ALP"]>Upper_limit]


# In[55]:


# After remove outliers check data not normally distributed and graph shown multipeak.
plt.figure(figsize=(20,20))
sns.histplot(data1,x="ALP",hue="Target",kde=True)
plt.grid()
plt.xlabel("ALP",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.legend(['Liver disease','Non liver disease'])
plt.show()


# ### Alamine_Aminotransferase

# In[56]:


data1.ALT.unique()


# In[57]:


data1.ALT.value_counts()


# In[58]:


plt.figure(figsize=(20,9))
sns.boxplot(x="ALT",
           showmeans=True,
           color='r',
           data=data1)


# In[59]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="ALT",hue="Target",kde=True)
plt.grid()
plt.xlabel("ALT",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# * Alamine aminotransferase are not normaly distributed and skewed right side.

# In[60]:


IQR=stats.iqr(data1.ALT,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.ALT.quantile(0.25)
print("Q1:",Q1)

Q3=data1.ALT.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-0.4*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+2.5*IQR
print("Upper limit:",Upper_limit)


# In[61]:


o=data1.loc[data1["ALT"]<Lower_limit]
o


# In[62]:


p=data1.loc[data1["ALT"]>Upper_limit]
p


# In[63]:


# Total outliers
ALT_out=(len(o+p)/len(data1))*100
print("Total outliers in ALT:{:.2f}%".format(ALT_out))

data1.drop(data1[data1["ALT"]>Upper_limit].index,inplace=True)
# In[64]:


data1.loc[data1["ALT"]>Upper_limit,"ALT"]=data1["ALT"].mean()


# In[65]:


data1.loc[data1["ALT"]>Upper_limit]


# ### Aspartate_Aminotransferase

# In[66]:


data1.AST.unique()


# In[67]:


data1.AST.value_counts()


# In[68]:


plt.figure(figsize=(20,9))
sns.boxplot(x="AST",
           showmeans=True,
           color='r',
           data=data1)


# In[69]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="AST",hue="Target",kde=True)
plt.grid()
plt.xlabel("Aspartate_Aminotransferase",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[70]:


IQR=stats.iqr(data1.AST,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.AST.quantile(0.25)
print("Q1:",Q1)

Q3=data1.AST.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-0.4*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+3*IQR
print("Upper limit:",Upper_limit)


# In[71]:


c=data1.loc[data1["AST"]<Lower_limit]
c


# In[72]:


d=data1.loc[data1["AST"]>Upper_limit]
d


# In[73]:


AST_out=(len(c+d)/len(data1))*100
print("Total Outliers of AST: {:.2f}%".format(AST_out))


# In[74]:


data1.loc[data1["AST"]>Upper_limit,"AST"]=data1["AST"].mean()


# In[75]:


data1.loc[data1["AST"]>Upper_limit]


# ### Total_Protiens

# In[76]:


data1.T_Protiens.unique()


# In[77]:


# length of value.
len(data1.T_Protiens.value_counts())


# In[78]:


plt.figure(figsize=(20,9))
sns.boxplot(x="T_Protiens",
           showmeans=True,
           color='r',
           data=data1)


# In[79]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="T_Protiens",hue="Target",kde=True)
plt.grid()
plt.xlabel("Total_Protiens",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# * As per graph normaly distributed all data

# In[80]:


IQR=stats.iqr(data1.T_Protiens,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.T_Protiens.quantile(0.25)
print("Q1:",Q1)

Q3=data1.T_Protiens.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-2*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper limit:",Upper_limit)


# In[81]:


e=data1.loc[data1["T_Protiens"]<Lower_limit]
e


# In[82]:


f=data1.loc[data1["T_Protiens"]>Upper_limit]
f


# In[83]:


T_out=((len(e+f))/len(data1))*100
print("Total Outliers of Total_protiens: {:.2f}%".format(T_out))


# * Very negligible amount of outliers so it's ignore
data1.loc[data1["T_Protiens"]<Lower_limit,"T_Protiens"]=data1["T_Protiens"].mean()data1.loc[data1["T_Protiens"]>Upper_limit,"T_Protiens"]=data1["T_Protiens"].mean()data1.loc[data1["T_Protiens"]<Lower_limit]data1.loc[data1["T_Protiens"]>Upper_limit]
# ### Albumin 

# In[84]:


data1.Albumin.unique()


# In[85]:


# Total length.
len(data1.Albumin.value_counts())


# In[86]:


plt.figure(figsize=(20,9))
sns.boxplot(x="Albumin",
           showmeans=True,
           color='r',
           data=data1)


# In[87]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="Albumin",hue="Target",kde=True)
plt.grid()
plt.xlabel("Albumin",fontsize=20)
plt.ylabel("Count",fontsize=20)
plt.show()


# In[88]:


IQR=stats.iqr(data1.Albumin,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.Albumin.quantile(0.25)
print("Q1:",Q1)

Q3=data1.Albumin.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-0.9*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+1.5*IQR
print("Upper limit:",Upper_limit)


# In[89]:


g=data1.loc[data1["Albumin"]<Lower_limit]
g

data1.loc[data1["Albumin"]<Lower_limit,"Albumin"]=data1["Albumin"].mean()data1.loc[data1["Albumin"]<Lower_limit]
# In[90]:


h=data1.loc[data1["Albumin"]>Upper_limit]
h


# * Outliers removed from features.

# ### Albumin_Globulin_Ratio

# In[91]:


data1.A_G.unique()


# In[92]:


data1.A_G.value_counts()


# In[93]:


plt.figure(figsize=(20,10))
sns.boxplot(x="A_G",
            showmeans=True,
            color="r",
            data=data1)


# In[94]:


plt.figure(figsize=(20,10))
sns.histplot(data1,x="A_G",kde=True,hue="Target")
plt.grid()
plt.xlabel("Albumin_Globulin_Ratio")
plt.ylabel("Count")
plt.plot


# In[95]:


IQR=stats.iqr(data1.A_G,interpolation="midpoint")
print("IQR:",IQR)

Q1=data1.A_G.quantile(0.25)
print("Q1:",Q1)

Q3=data1.A_G.quantile(0.75)
print("Q3:",Q3)

Lower_limit=Q1-0.5*IQR
print("Lower limit:",Lower_limit)

Upper_limit=Q3+4.5*IQR
print("Upper limit:",Upper_limit)


# In[96]:


i=data1.loc[data1["A_G"]<Lower_limit]
i


# In[97]:


j=data1.loc[data1["A_G"]>Upper_limit]
j


# In[98]:


A_G_out=(len(i+j)/len(data1))*100
print("Total outliers of A_G: {:.2f}%".format(A_G_out))


# In[99]:


data1.loc[data1["A_G"]<Lower_limit,"A_G"]=data1["A_G"].median()


# In[100]:


data1.loc[data1["A_G"]<Lower_limit]


# # Step:- 6 Feature Selection

# In[101]:


data1.isna().sum()


# In[102]:


data1.describe()


# In[103]:


data1.corr()


# In[104]:


plt.figure(figsize=(20,20))
sns.heatmap(data1.corr(),annot=True,cmap="copper_r")


# * Total_bilirubin is combination of direct bilirubin and indirect bilirubin so direct bilirubin features drop then not getting any impact to predict liver disease.
# Average of AST and ALT because impact of both are same. 
data1["Avg_bilirubin"]=(data1["Total_Bilirubin"]+data1["Direct_Bilirubin"])/2
# In[105]:


data1.drop("DRT_BL",axis=1,inplace=True)


# In[106]:


plt.figure(figsize=(20,20))
sns.heatmap(data1.corr(),annot=True,cmap="copper_r")
plt.show()


# # Task 2:-Create a predictive model with implementation of different classifiers on liver patient diseases dataset to predict liver diseases.

# ## Step 7:- Model selection and Building

# ## Splitting data

# In[107]:


data1


# In[108]:


data1.describe()


# In[109]:


X=data1.drop("Target",axis=1)


# In[110]:


y=data1.Target


# In[111]:


print(bold+"After splitting data:-\n"+reset,X)
print("\n")
print(bold+"After splitting data:-\n"+reset,y)


# In[112]:


y.value_counts()


# * Data is imbalance so need balacing of data.

# ### Model Building

# In[113]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=30,stratify=y)
print("X_train dataset shape:-",X_train.shape)
print("X_test dataset length:-",X_test.shape)
print("y_train dataset length:-",y_train.shape)
print("y_test dataset length:-",y_test.shape)


# In[114]:


X_train.isna().sum()


# ### Feature scaling of training and testing data

# In[116]:


# Copy of datasets
X_train_std=X_train.copy()
X_test_std=X_test.copy()

#Numerical features
num_columns=["TL_BL","ALP","ALT","AST","T_Protiens","Albumin","A_G"]

scale=StandardScaler()
# apply standardize on numerical columns
for i in num_columns:
    # Fit of training dataset
    std=scale.fit(X_train_std[[i]])
    # Transform of training dataset
    X_train_std[i]=std.transform(X_train_std[[i]])
    # Transform of testing dataset
    X_test_std[i]=scale.transform(X_test_std[[i]])
    


# In[117]:


# Convert dataset nd.array to dataframe
X_train_std_data=pd.DataFrame(X_train_std)
describ=X_train_std_data.describe()
X_train_std_data.describe()


# In[118]:


X_test_std.describe()


# ### Data Balancing with SMOTE method

# In[119]:


smote=SMOTE()


# In[120]:


X_smote,y_smote=smote.fit_resample(X_train_std_data,y_train)


# In[121]:


print(bold+"Actual classes:-"+reset,Counter(y_train))
print(bold+"SMOTE classes:-"+reset,Counter(y_smote))


# In[122]:


X_smote


# In[123]:


y_smote


# ## Logistic Regression

# In[124]:


# Train Logistic regression.
LRC=LogisticRegression()
LRC_data=LRC.fit(X_train_std_data,y_train)
LRC_data


# In[125]:


y_train_pred_LRC=LRC.predict(X_train_std_data)


# In[126]:


y_test_pred_LRC=LRC.predict(X_test_std)


# In[127]:


y_test_pred_LRC


# # Evalution of Logistic Regression

# In[128]:


print(classification_report(y_train_pred_LRC,y_train))


# In[129]:


# Accuracy score
print("Model training accuracy score: {:.2f}%".format(accuracy_score(y_train_pred_LRC,y_train)*100))
# Recall score
print("Model training recall score: {:.2f}%".format(recall_score(y_train_pred_LRC,y_train)*100))
# F1 score
print("Model training f1 score: {:.2f}%".format(f1_score(y_train_pred_LRC,y_train)*100))


# In[130]:


pd.crosstab(y_train_pred_LRC,y_train)


# In[131]:


print(classification_report(y_test_pred_LRC,y_test))


# In[132]:


pd.crosstab(y_test_pred_LRC,y_test)


# In[133]:


# Accuracy score
print("Model testing accuracy score: {:.2f}%".format(accuracy_score(y_test_pred_LRC,y_test)*100))
# Recall score
print("Model testing recall score: {:.2f}%".format(recall_score(y_test_pred_LRC,y_test)*100))
# F1 score
print("Model testing f1 score: {:.2f}%".format(f1_score(y_test_pred_LRC,y_test)*100))
# ROC_AUC score
print("Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test_pred_LRC,y_test)*100))


# In[134]:


fpr,tpr,thres=roc_curve(y_test_pred_LRC,y_test)
plt.plot(fpr,tpr,label="Logistic Regression")
plt.xlabel("False positive regression")
plt.ylabel("True positive regression")
plt.show


# ## Logistic regression using SMOTE techniques

# In[135]:


LRC_SMOTE=LogisticRegression()
LRC_SMOTE.fit(X_smote,y_smote)


# In[136]:


y_train_pred_LRC_smote=LRC_SMOTE.predict(X_train_std_data)


# In[137]:


print(classification_report(y_train_pred_LRC_smote,y_train))


# In[138]:


print("Training accuracy score: {:.2f}%".format(accuracy_score(y_train_pred_LRC_smote,y_train)*100))
print("Training recall score:{:.2f}%".format(recall_score(y_train_pred_LRC_smote,y_train)*100))
print("Training f1 score:{:.2f}%".format(f1_score(y_train_pred_LRC_smote,y_train)*100))
print("Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_train_pred_LRC_smote,y_train)*100))


# In[139]:


y_test_pred_LRC_smote=LRC_SMOTE.predict(X_test_std)


# In[140]:


print(classification_report(y_test_pred_LRC_smote,y_test))


# In[141]:


pd.crosstab(y_test_pred_LRC_smote,y_test)


# In[142]:


print("Testing Accuracy score:{:.2f}%".format(accuracy_score(y_test_pred_LRC_smote,y_test)*100))
print("Testing recall score:{:.2f}%".format(recall_score(y_test_pred_LRC_smote,y_test)*100))
print("Testing f1 score:{:.2f}%".format(f1_score(y_test_pred_LRC_smote,y_test)*100))
print("Testing roc_auc score:{:.2f}%".format(roc_auc_score(y_test_pred_LRC_smote,y_test)*100))


# In[143]:


fpr,tpr,thres=roc_curve(y_test_pred_LRC_smote,y_test)
plt.plot(fpr,tpr,label="Logistic Regression")
plt.xlabel("False positive regression")
plt.ylabel("True positive regression")
plt.show


# # KNN

# In[144]:


train_score={}
test_score={}
n_neighbors=np.arange(2,30,1)
for neighbors in n_neighbors:
    KNN_model=KNeighborsClassifier(n_neighbors=neighbors)
    KNN_model.fit(X_train_std_data,y_train)
    train_score[neighbors]=KNN_model.score(X_train_std_data,y_train)
    test_score[neighbors]=KNN_model.score(X_test_std,y_test)


# In[145]:


plt.figure(figsize=(15,10))
plt.plot(n_neighbors,train_score.values(),label="train accuracy")
plt.plot(n_neighbors,test_score.values(),label="test accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.title("KNN:Varying number of Neighbors")
plt.show()


# In[146]:


# With help of graph kvalue is 15 getting highest accuracy value.
for key,value in test_score.items():
    if value==max(test_score.values()):
        print("Highest accuracy value in KNeighbours value:-",key)


# In[147]:


KNN_model=KNeighborsClassifier(n_neighbors=7) # KNN classifier
KNN_data=KNN_model.fit(X_train_std_data,y_train) # Train model
KNN_data


# In[148]:


# predict training data.
y_train_predict=KNN_model.predict(X_train_std_data) 
# Predict testing data.
y_test_predict=KNN_model.predict(X_test_std)


# ### Model Evalution

# In[149]:


# Classification report for training data.
print(classification_report(y_train_predict,y_train))


# In[150]:


# Accuracy score
print("Model training accuracy score:- {:.2f}%".format(accuracy_score(y_train_predict,y_train)*100))
# Recall score
print("Model training recall score:- {:.2f}%".format(recall_score(y_train_predict,y_train)*100))
# F1 score
print("Model training f1 score:- {:.2f}%".format(f1_score(y_train_predict,y_train)*100))
# roc auc score
print("Model training roc_auc score:- {:.2f}%".format(roc_auc_score(y_train_predict,y_train)*100))


# In[151]:


# Classification report for testing data.
print(classification_report(y_test_predict,y_test))


# In[152]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_test_predict,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_test_predict,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_test_predict,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_test_predict,y_test)*100))


# In[153]:


pd.crosstab(y_train_predict,y_train)


# In[154]:


pd.crosstab(y_test_predict,y_test)


# In[155]:


fpr,tpr,thres=roc_curve(y_test_predict,y_test)
plt.plot(fpr,tpr,label='KNeighbors Classifier')
plt.xlabel("false positive regression")
plt.ylabel("false negative regression")
plt.show()


# ## Hyperparameter Tuning

# In[156]:


kf=KFold(n_splits=5,shuffle=True,random_state=40)
parameter={'n_neighbors':np.arange(2,30,1)}
Knn=KNeighborsClassifier()
Knn_cv=GridSearchCV(Knn,param_grid=parameter,cv=kf,verbose=1)
Knn_cv.fit(X_train_std_data,y_train)
print(Knn_cv.best_params_)


# In[157]:


# predict training data.
y_train_predict=KNN_model.predict(X_train_std_data) 
# Predict testing data.
y_test_predict=KNN_model.predict(X_test_std)


# In[158]:


print(classification_report(y_test_predict,y_test))


# In[159]:


pd.crosstab(y_test_predict,y_test)


# ### Balancing dataset with KNN SMOTE techniques

# In[160]:


# KNN classifier with balance dataset
KNN_model_SMOTE=KNeighborsClassifier(n_neighbors=2)
# Train model
KNN_data_SMOTE=KNN_model_SMOTE.fit(X_smote,y_smote)
KNN_data_SMOTE


# In[161]:


# Prediction of training data
y_training_data_SMOTE_predict=KNN_data_SMOTE.predict(X_train_std_data)
# Prediction of testing
y_test_data_SMOTE_predict=KNN_data_SMOTE.predict(X_test_std)


# ### Model Evalution

# In[162]:


print(classification_report(y_training_data_SMOTE_predict,y_train))


# In[163]:


# Accuracy score
print("Model training accuracy score: {:.2f}%".format(accuracy_score(y_training_data_SMOTE_predict,y_train)*100))
# Recall score
print("Model training recall score: {:.2f}%".format(recall_score(y_training_data_SMOTE_predict,y_train)*100))
# F1 score
print("Model training f1 score: {:.2f}%".format(f1_score(y_training_data_SMOTE_predict,y_train)*100))
# roc auc score
print("Model training roc_auc score: {:.2f}%".format(roc_auc_score(y_training_data_SMOTE_predict,y_train)*100))


# In[164]:


print(classification_report(y_test_data_SMOTE_predict,y_test))


# In[165]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_test_data_SMOTE_predict,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_test_data_SMOTE_predict,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_test_data_SMOTE_predict,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_test_data_SMOTE_predict,y_test)*100))


# In[166]:


pd.crosstab(y_test_data_SMOTE_predict,y_test)


# In[167]:


fpr,tnr,thres=roc_curve(y_test_data_SMOTE_predict,y_test)
plt.plot(fpr,tpr,label="KNeighbiurs regression")
plt.xlabel=("false postive regression")
plt.ylabel("false negative regression")
plt.show()


# # Decision Tree

# In[168]:


# buid decision tree model.
tree=DecisionTreeClassifier()
# Train model
tree.fit(X_train_std_data,y_train)


# In[169]:


# Predict training data.
y_train_pred_DT=tree.predict(X_train_std_data)
# Predict testing data
y_test_pred_DT=tree.predict(X_test_std)


# ### Model Evalution

# In[170]:


print(classification_report(y_train_pred_DT,y_train))   


# In[171]:


pd.crosstab(y_train_pred_DT,y_train)


# In[172]:


print(classification_report(y_test_pred_DT,y_test))


# In[173]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_test_pred_DT,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_test_pred_DT,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_test_pred_DT,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_test_pred_DT,y_test)*100))


# In[174]:


pd.crosstab(y_test_pred_DT,y_test)


# ### Using Hyper parameter tuning

# In[ ]:


DT_clf= DecisionTreeClassifier(random_state=4) # Object create decision tree with random state.
param={
       "criterion":("gini","entropy"),   # Decide measure of quality of split based on criteria
       "splitter":("best","random"),     #  
       "max_depth":(list(range(1,30))),  # Max depth of tree
       "min_samples_split":[2,3,4],       # Minimum number of sample required to slit node
       "min_samples_leaf":(list(range(1,30)))  # Min number of sample required for leaf node.
       }
DT_cv=GridSearchCV(DT_clf,param,scoring="accuracy",verbose=1,cv=3)
# DT_clf= Model for training.
# param= hyperparameters (Dictonary created)
# scoring= performance matrix to performance checking.
# verbose= control the verbosity; the more message.
#>1=the computation time for each fold and parameter candidate is displayed;
#>2= the score is also displayed;
#>3= the fold and candidate parameter indexes are also displayed together with the starting time of the computation.
#cv= number of flods

DT_cv.fit(X_train_std_data,y_train)   # To training of gridsearch cv.
best_params=DT_cv.best_params_    # Give you best parameters.
print(f"Best parameters: {best_params}")


# In[ ]:


# Impute best parameters
DT_clff=DecisionTreeClassifier(criterion= 'gini', max_depth= 6, min_samples_leaf=19, min_samples_split= 2, splitter='best')


# In[ ]:


DT_clff1=DT_clff.fit(X_train_std_data,y_train)
# Predict data
y_pred_DT=DT_clff.predict(X_test_std)
y_pred_DT


# ## Evalution of Decision tree using hyperparameter tuning

# In[ ]:


print(classification_report(y_pred_DT,y_test))


# In[ ]:


# Recall score in percentage
Test1=(recall_score(y_pred_DT,y_test)*100)
print("Recall score: {:.2f}%".format(Test1))
# F1 score in percentage
Test2=(f1_score(y_pred_DT,y_test)*100)
print("f1 score: {:.2f} % ".format(Test2))
# Accuracy score in percentage
Test3=(accuracy_score(y_pred_DT,y_test)*100)
print("Accuracy score: {:.2f}%".format(Test3))
# Roc_auc score in percentage
Test4=(roc_auc_score(y_pred_DT,y_test)*100)
print("ROC_AUC score:{:.2f}%".format(Test4))


# In[ ]:


pd.crosstab(y_pred_DT,y_test)


# # Decision Tree with SMOTE techniques

# In[ ]:


# Fit decision tree with data balancing.
tree.fit(X_smote,y_smote)


# In[ ]:


# Predict training data
y_training_DT_SMOTE_predict=tree.predict(X_train_std_data)
# Predict testing data
y_test_DT_SMOTE_predict=tree.predict(X_test_std)


# ## Evalution of Decision tree using SMOTE

# In[ ]:


print(classification_report(y_training_DT_SMOTE_predict,y_train))


# In[ ]:


print(classification_report(y_test_DT_SMOTE_predict,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_test_DT_SMOTE_predict,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_test_DT_SMOTE_predict,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_test_DT_SMOTE_predict,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_test_DT_SMOTE_predict,y_test)*100))


# In[ ]:


pd.crosstab(y_test_DT_SMOTE_predict,y_test)


# # Random forest

# In[ ]:


# build random forest model
rf_clf=RandomForestClassifier()
# train model
rf_clf.fit(X_train_std_data,y_train)


# In[ ]:


# Predict training data
y_prd_training_data_rf=rf_clf.predict(X_train_std_data)
# Prdict testing data
y_prd_testing_data_rf=rf_clf.predict(X_test_std)


# ## Evalution of Model

# In[ ]:


print(classification_report(y_prd_testing_data_rf,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_prd_testing_data_rf,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_prd_testing_data_rf,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_prd_testing_data_rf,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_prd_testing_data_rf,y_test)*100))


# In[ ]:


pd.crosstab(y_prd_testing_data_rf,y_test)


# ## Random forest with smote techniques

# In[ ]:


# build random forest model
rf_clf=RandomForestClassifier()
# train model
rf_clf.fit(X_smote,y_smote)


# In[ ]:


# Predict training data
y_prd_training_rf_smote=rf_clf.predict(X_train_std_data)
# Prdict testing data
y_prd_testing_rf_smote=rf_clf.predict(X_test_std)


# ## Evalution of SMOTE techniques

# In[ ]:


print(classification_report(y_prd_testing_rf_smote,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(y_prd_testing_rf_smote,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(y_prd_testing_rf_smote,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(y_prd_testing_rf_smote,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(y_prd_testing_rf_smote,y_test)*100))


# In[ ]:


pd.crosstab(y_prd_testing_rf_smote,y_test)


# ## Hyperparameter tuning of Random Forest

# In[ ]:


# Random search used setup a grid of hyperparameter value and selects random combination of train the model and score.
# Model 
rf_classi=RandomForestClassifier(random_state=40)

criterion=['gini','entropy']
n_estimators= [int(x) for x in np.linspace(start=100,stop=1000,num=15)] # Number of trees used random forest.
max_features= ['auto','sqrt','log2'] # Maximum number of features allowed to try in individual tree
max_depth=[int(x) for x in np.linspace(start=2,stop=90,num=15)]  # Maximum number of depth of iteration
min_sample_split= [2,5,8,10,12]  # Minimum no of sample split 
min_sample_leaf= [1,2,3,4,5,6]  # Minimum number of sample of leaf split
bootstrap= [True,False] # Sampling 

#dictionary for hyperparameters
random_grid={'n_estimators':n_estimators,'max_features':max_features,'max_depth':max_depth,
             'min_samples_split':min_sample_split,'min_samples_leaf':min_sample_leaf,'bootstrap':bootstrap}

rf_clf_hyp= RandomizedSearchCV(estimator=rf_classi,scoring='f1',param_distributions=random_grid,n_iter=100,cv=2,
                              verbose=1,n_jobs=-1)
# build training model
rf_clf_hyp.fit(X_train_std_data,y_train)
# Best parameter 
rf_best_param=rf_clf_hyp.best_params_

print(f'Best parameters: {rf_best_param}')


# In[ ]:


# Impute best parameter
rf_clf_param=RandomForestClassifier(n_estimators=807,min_samples_split= 8,min_samples_leaf= 5,max_features= 'auto',max_depth= 2,bootstrap=True)


# In[ ]:


# Model fit to data
rf_clf_param=rf_clf_param.fit(X_train_std_data,y_train)
# predict data with training data
rf_clf_training_pred_param=rf_clf_param.predict(X_train_std_data)
# Predict data with testing data
rf_clf_test_pred_param=rf_clf_param.predict(X_test_std)
rf_clf_test_pred_param


# ## Evalution of Hyperparameter tuning RF

# In[ ]:


print(classification_report(rf_clf_training_pred_param,y_train))


# In[ ]:


print(classification_report(rf_clf_test_pred_param,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(rf_clf_test_pred_param,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(rf_clf_test_pred_param,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(rf_clf_test_pred_param,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(rf_clf_test_pred_param,y_test)*100))


# In[ ]:


pd.crosstab(rf_clf_test_pred_param,y_test)


# # XGBoost

# In[ ]:


# Model creation
xgb_creation=XGBClassifier()
# Model build
xgb_creation.fit(X_train_std_data,y_train)


# In[ ]:


# Predict training data
xgb_traing_pred=xgb_creation.predict(X_train_std_data)
print("Traing data predict:- \n",xgb_traing_pred)
# Predict testing data
xgb_test_pred=xgb_creation.predict(X_test_std)
print("Testing data predict:- \n",xgb_test_pred)


# ## Evalution of XGBoost Model

# In[ ]:


print(classification_report(xgb_test_pred,y_test))


# In[ ]:


print("Testing data accuracy score: {:.2f}%".format(accuracy_score(xgb_test_pred,y_test)*100))
print("Testing data recall score: {:.2f}%".format(recall_score(xgb_test_pred,y_test)*100))
print("Testing data f1 score: {:.2f}%".format(f1_score(xgb_test_pred,y_test)*100))
print("Testing data AUC_ROC score: {:.2f}%".format(roc_auc_score(xgb_test_pred,y_test)*100))


# In[ ]:


pd.crosstab(xgb_test_pred,y_test)


# ## XGBOOST with SMOTE techniques

# In[ ]:


# Model creation
xgboost_smote=XGBClassifier()
# Model build 
xgboost_smote.fit(X_smote,y_smote)


# In[ ]:


# predict training and testing data
xgb_training_pred_smote_data=xgboost_smote.predict(X_train_std_data)
xgb_testing_pred_smote_data=xgboost_smote.predict(X_test_std)


# ## Evalution of XGBOOST SMOTE data

# In[ ]:


print(classification_report(xgb_training_pred_smote_data,y_train))


# In[ ]:


print(classification_report(xgb_testing_pred_smote_data,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(xgb_testing_pred_smote_data,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(xgb_testing_pred_smote_data,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(xgb_testing_pred_smote_data,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(xgb_testing_pred_smote_data,y_test)*100))


# In[ ]:


pd.crosstab(xgb_testing_pred_smote_data,y_test)


# ## Hyperparameter tuning of XGBOOST model

# In[ ]:


# Random state
XGBoost_classifier=XGBClassifier(silent=0,random_state=40)
# Hyperparameter
best_param_xgb={'learning_rate': [0.01, 0.03, 0.06, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7,0.9],
                'max_depth': [2,3,4,5,7,9,10,12,14,15],
                #'grow_policy':[0,1],
                'max_leaves':[2,3,4,5,6,7,8,9,10],
                'verbosity':[0,1,2,3],
                'gamma':[0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150],
                'n_estimators':[10,20,30,50,65,80,100,115,130,150],
                'reg_alpha': [0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150],
                'reg_lambda': [0,0.1,0.2,0.3,0.9,1.6,3.9,6.4,18.4,26.7,54.6,110.8,150]
                }

XGB_rcv=RandomizedSearchCV(estimator=XGBoost_classifier,scoring='f1',param_distributions=best_param_xgb,
                   n_iter=100,n_jobs=-1,cv=3,random_state=50,verbose=3)
# Training data of randomizedsearchcv
XGB_rcv.fit(X_train_std_data,y_train)

# Best parameters
XGB_best_param=XGB_rcv.best_params_
print(f"best parameters:{XGB_best_param}")


# In[ ]:


XGB_best_param=XGBClassifier(verbosity= 0,reg_lambda=110.8,reg_alpha=0.2,n_estimators=50,max_leaves=8,max_depth=7,learning_rate=0.1,gamma= 0.2)


# In[ ]:


# fit with data
XGB_data_best_param=XGB_best_param.fit(X_train_std_data,y_train)
# Training data predict
XGB_training_data_best_param=XGB_data_best_param.predict(X_train_std_data)
# Testing data predict
XGB_testing_data_best_param=XGB_data_best_param.predict(X_test_std)


# ## Evalution of Hyperparameter tuning RF

# In[ ]:


print(classification_report(XGB_training_data_best_param,y_train))


# In[ ]:


print(classification_report(XGB_testing_data_best_param,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(XGB_testing_data_best_param,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(XGB_testing_data_best_param,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(XGB_testing_data_best_param,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(XGB_testing_data_best_param,y_test)*100))


# In[ ]:


pd.crosstab(XGB_testing_data_best_param,y_test)


# # Support vector Machine

# In[ ]:


# model build
svc=SVC()
# Model fit with data
svc.fit(X_train_std_data,y_train)


# In[ ]:


# predict training data and testing data
SVC_training_prd=svc.predict(X_train_std_data)
SVC_testing_prd=svc.predict(X_test_std)


# ## Evalution of SVM 

# In[ ]:


print(classification_report(SVC_training_prd,y_train))


# In[ ]:


print(classification_report(SVC_testing_prd,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(SVC_testing_prd,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(SVC_testing_prd,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(SVC_testing_prd,y_test)*100))


# In[ ]:


pd.crosstab(SVC_testing_prd,y_test)


# ## SVM with SMOTE

# In[ ]:


svc_smote=SVC()
svc_smote.fit(X_smote,y_smote)


# In[ ]:


svc_training_smote_pred=svc_smote.predict(X_train_std_data)
svc_testing_smote_pred=svc_smote.predict(X_test_std)


# ## Evalution of SVM smote

# In[ ]:


print(classification_report(svc_training_smote_pred,y_train))


# In[ ]:


print(classification_report(svc_testing_smote_pred,y_test))


# In[ ]:


# Accuracy score
print("Model testing accuracy score:- {:.2f}%".format(accuracy_score(svc_testing_smote_pred,y_test)*100))
# Recall score
print("Model testing recall score:- {:.2f}%".format(recall_score(svc_testing_smote_pred,y_test)*100))
# F1 score
print("Model testing f1 score:- {:.2f}%".format(f1_score(svc_testing_smote_pred,y_test)*100))
# roc auc score
print("Model testing roc_auc score:- {:.2f}%".format(roc_auc_score(svc_testing_smote_pred,y_test)*100))


# In[ ]:


pd.crosstab(svc_testing_smote_pred,y_test)


# # Task3:- Create an analysis to show on what basis you have designed your model.  

# ## Summary:-

# * Logistic regression accuracy with imbalance data=72%, recall score=73.33%, f1 score= 82.80%
# * Logistic regression accuracy with balance data=72%, recall score=86.76%, f1 score=79.19%
# 

# * KNN algorithm with imbalance data accuracy score=70.18%, recall score=71.17%, f1 score=82.29%
# * KNN algorithm with balanced data accuracy score=62%, recall score=88%, f1 score=67% 
# * Conclusion is that with balanced dataset FNR value decrease so good model for us but accuracy decrease.

# * Decision tree algorithm with imbalance data accuracy score=66.67%,recall score=74.71%, f1 score=77.38%
# * Decision tree algorithms with hyperparameter tuning data accuracy score=66.67%,recall score=73.61%, f1 score=77.91%
# * Decision tree algorithms with balanced data accuracy score=71.05%, recall score=78.57%, f1 score=80%
# * conclusion is that when use balanced dataset then decrease FNR value and increase f1 score, accuracy also increase.

# * Random forest algorithm with imbalance data accuracy score=69.30%,recall score=77.38%, f1 score=78.79%
# * Random forest algorithms with hyperparameter tuning data accuracy score=72.81%,recall score=72.32%, f1 score=83.94%
# * Random forest algorithms with balanced data accuracy score=69.30%,recall score=77.38%, f1 score=78.79%
# * Conclusion is that with help of hyperparameter tuning data accuracy is good but FNR value increase that's not good more patient prediction model.

# * XGBoost algorithm with imbalance data accuracy score=73.68%,recall score=77.42%, f1 score=82.68%
# * XGBoost forest algorithms with hyperparameter tuning data accuracy score=75.44%,recall score=75.24%, f1 score=84.95%
# * XGBoost forest algorithms with balanced data accuracy score=78.09%,recall score=82.56%, f1 score=85.03%
# * Conclusion is that with help of balanced dataset getting good accuracy and FNR value decrease & f1 score also increase  that's good more patient prediction model.

# * In SVM algorithm getting only 70% accuracy, recall score=77.65% and f1 score=79.52% 

# # Conclussion:-

# ### As per above summary we define that XGBoost model is good for liver disease patient prediction.

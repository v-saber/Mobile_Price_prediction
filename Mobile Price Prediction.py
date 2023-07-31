#!/usr/bin/env python
# coding: utf-8

# <div style=" font-size:22px; line_height:160%">
# importing libraries

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import metrics
from collections import Counter
# import warnings


# In[2]:


# warnings.simplefilter(action="ignore")


# <div style=" font-size:14px; line_height:160%">
# dispaly all of the columns

# In[3]:


pd.set_option("display.max_columns", None)


# <div style=" font-size:14px; line_height:160%">
# improting the datasets of train and test

# In[4]:


mobile_data_train = pd.read_csv("D:\programming\Machine Learning/mobile_train_dataset.csv")
mobile_data_test = pd.read_csv("D:\programming\Machine Learning/mobile_test_dataset.csv")


# <div style=" font-size:14px; line_height:160%">
# showing the top 5 rows of the datasets

# In[5]:


mobile_data_train.head(5)


# In[6]:


mobile_data_test.head(5)


# <div style=" font-size:18px; line_height:160%">
# the last column - "price_range" - is our target

# <div style=" font-size:14px; line_height:160%">
# making a copy of the train dataset

# In[7]:


df = mobile_data_train.copy()

df.shape


# In[8]:


df


# <div style=" font-size:14px; line_height:160%">
# in the cell below we see the analytical information of the dataset

# In[9]:


df.describe(include="all")


# <div style=" font-size:14px; line_height:160%">
# to check if the dataset has NaN values and also to see the type of each feature

# In[10]:


df.info()


# <div style=" font-size:14px; line_height:160%">
# We don't have missing values.

# <div style=" font-size:14px; line_height:160%">
# to see how many unique values are in each column:

# In[11]:


df.nunique()


# <div style=" font-size:14px; line_height:160%">
# to find out if there are any duplicate rows in the dataset:

# In[12]:


df.index.duplicated().sum()


# <div style=" font-size:14px; line_height:160%">
# there is no duplicated rows in the dataset.

# <div style=" font-size:14px; line_height:160%">
# drawing the correlation diagram:

# In[13]:


plt.figure(figsize=(20,10), dpi=80)
sns.heatmap(df.corr(), annot=True, cmap=plt.cm.Blues)
plt.show()


# <div style=" font-size:14px; line_height:160%">
# we can say there is no significant correlation between the features.

# <div style=" font-size:14px; line_height:160%">
# defining a function to draw the scatter plots between the target and features:

# In[14]:


def scatter_plots(df_name, x_ax_name, y_ax_name):
    scatter_name = f"{y_ax_name}-{x_ax_name}"
    fig_output_name = scatter_name
    plt.title(f"{x_ax_name} - {y_ax_name}\n")
    scatter_name = plt.scatter(df_name[x_ax_name], df_name[y_ax_name])
    scatter_name.axes.tick_params(gridOn=True, size=12, labelsize=10)
    plt.xlabel(f"\n{x_ax_name}", fontsize=20)
    plt.ylabel(f"{y_ax_name}\n", fontsize=20)
    plt.xticks(rotation=90)


# <div style=" font-size:14px; line_height:160%">
# defining a function to draw scatter plots side by side:

# In[15]:


def scatter_subplots(df):
    i=0
    j = len(df.columns)-1
    while i < len(df.columns)-1:
        plt.figure(figsize=(20,8), dpi=80)
        for k in range(3): 
            plt.subplot(1, 3, k+1)
            scatter_plots(df, df.columns[i], df.columns[j])
            plt.title(f"{df.columns[i]} - {df.columns[j]}", fontsize=20)            
            i += 1
        plt.suptitle("Plotting Each Feature Against The Target", size = 30, fontweight = "bold")
        plt.tight_layout()
        plt.show()
        if j-i == 2:
            plt.subplot(1, 2, 1)
            scatter_plots(df, df.columns[i], df.columns[j])
            plt.title(f"{df.columns[i]} - {df.columns[j]}", fontsize=20)   
            i += 1
            plt.subplot(1, 2, 2)
            scatter_plots(df, df.columns[i], df.columns[j])
            plt.title(f"{df.columns[i]} - {df.columns[j]}", fontsize=20)
            i += 1
            plt.suptitle("Plotting Each Feature Against The Target", size = 30, fontweight = "bold")
            plt.tight_layout()
            plt.subplots_adjust(wspace=2.5)
            plt.show()
        elif j-i == 1:
            scatter_plots(df, df.columns[i], df.columns[j])
            plt.title(f"{df.columns[i]} - {df.columns[j]}", fontsize=20)
            i += 1
            plt.suptitle("Plotting Each Feature Against The Target", size = 30, fontweight = "bold")
            plt.tight_layout()
            plt.subplots_adjust(wspace=2.5)
            plt.show()
        elif i == j:
            break


# In[16]:


scatter_subplots(df)


# <div style=" font-size:14px; line_height:160%">
# defining a function to count the redundancy of categories in each feature:

# In[17]:


def count_plots(df_name, column_name):

    plt.figure(figsize=(20, 8), dpi=90)
    ax = sns.countplot(x=column_name, data=df)
    ax.bar_label(ax.containers[0], fontsize=13)
    plt.xticks(rotation=90, fontsize=12)
    plt.yticks(fontsize=15)
    plt.xlabel(f"\n{column_name}", fontsize=20)
    plt.ylabel("count",fontsize=20)
    plt.title(f"Count of {column_name}", fontsize=30)
    plt.grid()


# <div style=" font-size:14px; line_height:160%">
# drawing the count plots:

# In[18]:


i = 0
j = len(df.columns)-1
while i<=j:
    count_plots(df, df.columns[i])
    plt.grid()
    i += 1
    plt.tight_layout()
    plt.show()


# <div style=" font-size:14px; line_height:160%">
# count of each category in the target column:

# In[19]:


Counter(df["price_range"])


# <div style=" font-size:14px; line_height:160%">
# according to the dataset, there are samples with the "screen width" equal to zero! It is definitely impossible for a phone to have screen width = 0. Let's count them:

# In[20]:


Counter(df["sc_w"]==0)


# In[21]:


print(f"The pencentage of screen width = 0 is:  % {(180/2000)*100}")


# <div style=" font-size:14px; line_height:160%">
# Almost 10 percent of the samples have screen width = 0.

# In[22]:


scatter_plots(df, "sc_w", "sc_h")


# <div style=" font-size:14px; line_height:160%">
# Take a look at this plot. There are mobile phones with screen width less than 2 cm and screen height from 5 cm up to 19 cm!  Do we have such phones in the real world?!

# In[23]:


Counter(mobile_data_test["sc_w"]==0)


# <div style=" font-size:14px; line_height:160%">
# As we can see, there are almost 10 percent of "screen width = 0" in the samples of the test dataset too. So, maybe we shouldn't do anything about them and leave them the way they are now. The owner of the data must know something about it.

# In[24]:


Counter(df["m_dep"]>0.5)
# Counter(df["price_range"][df["m_dep"]>0.5])


# <div style=" font-size:14px; line_height:160%">
# The lowest thickness of a cellphone is greater than 0.5 mm. But here we see that over half of the samples in the dataset have mobile depth equal to 0.5 or lower. We absolutely need the data owner in this case.

# In[25]:


# sorting the dataset by pixel width and pixel height respectively.
df.sort_values(by=["px_height", "px_width"])


# In[26]:


Counter(df["px_height"]<320)


# <div style=" font-size:14px; line_height:160%">
# We have zeros in pixel height. Besides, we know the minimum pixel height for a mobile phone is 320 pixels. Here in the dataset we have 568 samples with pixel heights less than 320 pixels. 

# In[27]:


Counter(df["px_width"]<df["px_height"])


# In[28]:


Counter(df["px_height"]<df["px_width"])


# <div style=" font-size:14px; line_height:160%">
# On the other hand, generally in mobile phones the amount of pixel height is higher than the amount of pixel width. As it can be seen, in this dataset it is completely inverse! It seems that maybe we should swap the labels!

# In[29]:


df_num_touch = df[df["touch_screen"]==0]
df_num_touch.shape


# <div style=" font-size:14px; line_height:160%">
# 994 samples doesn't have touch screen which might not give us a significant sense about the datas.

# In[30]:


df_num_3G = df[df["three_g"]==0]
Counter(df_num_3G["four_g"])


# <div style=" font-size:14px; line_height:160%">
# There are 477 samples that doesn't have 3G or 4G .

# In[31]:


df_num_camera = df[df["pc"]==0]
Counter(df_num_camera["fc"])


# <div style=" font-size:14px; line_height:160%">
# There are 101 samples that doesn't have front or primary camera.

# In[32]:


df_num_front = df[df["fc"]==0]
Counter(df_num_front["pc"]>0)

# Counter(df_num_front["pc"])           # it gives us a dict, it's keys are "pc values" and it's values are "the counts of pc values"
# Counter(df_num_front["pc"]).values()  # it gives us "the counts of values" in a list and does't give us the "pc values"
# df_num_front["pc"].value_counts()     # it gives us the "pc values" and the "counts of each pc value" sorted by the counts
# sorted(Counter(df_num_front["pc"]))   # it gives us only the keys (the pc values) and not the count of each one


# <div style=" font-size:14px; line_height:160%">
# There are 373 samples that have primary camera but don't have front camera.

# In[33]:


df_num_blue = df[df["blue"]==0]
df_num_blue.shape


# <div style=" font-size:14px; line_height:160%">
# There are 1010 samples that doesn't have bluetooth.

# In[34]:


df_num_wifi = df[df["wifi"]==0]
df_num_wifi.shape


# <div style=" font-size:14px; line_height:160%">
# There are 986 samples that doesn't have wifi.

# <div style=" font-size:26px; line_height:160%">
# Let's setup our models

# <div style=" font-size:22px; line_height:160%">
# Decision Tree Model

# In[35]:


df


# In[36]:


X = df.drop("price_range", axis=1).values
y = df.price_range.values.reshape(-1, 1)
# X, y are now both 2D arrays


# In[37]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[38]:


dt = DecisionTreeClassifier(random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# In[39]:


dt_accuracy = metrics.accuracy_score(y_test, y_pred)


# In[40]:


print("Accuracy:", dt_accuracy)


# <div style=" font-size:14px; line_height:160%">
# finding the best hyperparameters

# In[41]:


parameters = {"max_depth": range(1, 20), 
              "splitter": ["best", "random"]
             }

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_dt = GridSearchCV(estimator=dt,  # Model
                      param_grid=parameters,
                      scoring="accuracy",  # Strategy to evaluate the performance of the cross-validated model on the test set
                                           # if it is a multiclass target, use f1_micro
                                           # f1 or roc_auc doesn't work with multiclass targets
                                           # f1_micro and accuracy were OK here.
                      cv=cv ,  #cross-validation generator
                      verbose=1,  #Time to calculate
                      n_jobs=-1)  #Help to CPU

grid_dt.fit(X_train, y_train.ravel())


# In[42]:


grid_dt.best_params_


# <div style=" font-size:14px; line_height:160%">
# building the model with the best hyperparameters

# In[43]:


dt = DecisionTreeClassifier(max_depth=8, 
                            splitter='best', 
                            random_state=0)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)


# <div style=" font-size:14px; line_height:160%">
# The Confusion Matrix

# In[44]:


cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=dt.classes_)
disp.plot()
plt.show()


# <div style=" font-size:14px; line_height:160%">
# The Classification Report

# In[45]:


target_names = ["class '0'", "class '1'", "class '2'", "class '3'"]
print(classification_report(y_test, y_pred, target_names=target_names))


# <div style=" font-size:18px; line_height:160%">
# Now we have to predict the prices of the given mobile phones features dataset.

# In[46]:


DF = mobile_data_test.copy()

DF.shape


# <div style=" font-size:14px; line_height:160%">
# The dataset has one extra column. Let's see the column labels:

# In[47]:


DF.columns


# <div style=" font-size:14px; line_height:160%">
# The label "id" is surplus and needs to be dropped.

# In[48]:


DF.drop("id", axis=1, inplace=True)


# <div style=" font-size:14px; line_height:160%">
# Let's predict the test data:

# In[49]:


dt.fit(X, y)

predicted_prices_dt = dt.predict(DF.values)
predicted_prices_dt = pd.DataFrame(predicted_prices_dt.reshape(-1, 1), columns=["price_range"])

mobile_prices = pd.concat([DF, predicted_prices_dt], axis=1)
mobile_prices


# <div style=" font-size:22px; line_height:160%">
# Random Forest

# In[50]:


X = df.drop("price_range", axis=1).values
y = df.price_range.values.reshape(-1, 1)
# X, y are now both 2D arrays


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# In[52]:


rf = RandomForestClassifier(random_state=0)
rf.fit(X_train, y_train.ravel())
y_pred = rf.predict(X_test)


# In[53]:


rf_accuracy = metrics.accuracy_score(y_test, y_pred)


# In[54]:


print("Accuracy:", rf_accuracy)


# In[55]:


parameters = [
              {"max_depth": range(5, 20), 
               "n_estimators": [50, 100, 150]
              }
             ]

cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)
grid_rf = GridSearchCV(estimator=rf,  # Model
                      param_grid=parameters, 
                      scoring="f1_micro",  # Strategy to evaluate the performance of the cross-validated model on the test set
                                           # if it is a multiclass target, use f1_micro
                                           # f1 or roc_auc doesn't work with multiclass targets
                                           # f1_micro and accuracy were OK here.
                      cv=cv,  # cross-validation generator
                      verbose=1,  # Time to calculate
                      n_jobs=-1)  # Help to CPU

grid_rf.fit(X_train, y_train.ravel())


# In[56]:


grid_rf.best_params_


# In[57]:


rf = RandomForestClassifier(n_estimators=150, 
                            max_depth=17, 
                            random_state=0)
rf.fit(X_train, y_train.ravel())
y_pred = rf.predict(X_test)


# In[58]:


cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
disp.plot()
plt.show()


# In[59]:


target_names = ["class '0'", "class '1'", "class '2'", "class '3'"]
print(classification_report(y_test, y_pred, target_names=target_names))


# <div style=" font-size:14px; line_height:160%">
# Let's predict the test data:

# In[60]:


rf.fit(X, y.ravel())

predicted_prices_rf = rf.predict(DF.values)
predicted_prices_rf = pd.DataFrame(predicted_prices_rf.reshape(-1, 1), columns=["price_range"])

mobile_prices_rf = pd.concat([DF, predicted_prices_rf], axis=1)
mobile_prices_rf


# <div style=" font-size:22px; line_height:160%">
# Support Vector Machines

# In[61]:


X = df.drop("price_range", axis=1).values
y = df.price_range.values.reshape(-1, 1)
# X, y are now both 2D arrays


# In[62]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# <div style=" font-size:14px; line_height:160%">
# Since this model works based on the distances between points, we have to scale the features' values.

# <div style=" font-size:14px; line_height:160%">
# We use Standard Scaler to scale the features' values.

# In[63]:


scaler = StandardScaler()
X_train_scaling = scaler.fit_transform(X_train)
X_test_scaling = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaling, columns=df.drop("price_range", axis=1).columns)
X_test_scaled = pd.DataFrame(X_test_scaling, columns=df.drop("price_range", axis=1).columns)


# In[64]:


svm = SVC(random_state=0)
svm.fit(X_train, y_train.ravel())
y_pred = svm.predict(X_test)


# In[65]:


svm_accuracy = metrics.accuracy_score(y_test, y_pred)


# In[66]:


print("Accuracy:", svm_accuracy)


# In[67]:


parameters = [{'kernel': ['poly'], 
               'degree': [2, 3, 4, 5], 
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
               'C': [0.01, 0.1, 1, 10, 100, 1000]},
                  
              {'kernel': ['rbf','sigmoid'],
               'gamma': [1, 0.1, 0.01, 0.001, 0.0001], 
               'C': [0.01, 0.1, 1, 10, 100, 1000]},
                  
              {'kernel': ['linear'],
               'C': [0.01, 0.1, 1, 10, 100, 1000]}
             ]

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
grid_svm = GridSearchCV(estimator=svm,  # Model
                      param_grid=parameters,  
                      scoring="f1_micro",  # Strategy to evaluate the performance of the cross-validated model on the test set
                                           # if it is a multiclass target, use f1_micro
                                           # f1 or roc_auc doesn't work with multiclass targets
                                           # f1_micro and accuracy were OK here.
                      cv=cv,  # cross-validation generator
                      verbose=1,  # Time to calculate
                      n_jobs=-1)  # Help to CPU

grid_svm.fit(X_train, y_train.ravel())


# In[68]:


grid_svm.best_params_


# In[69]:


svm = SVC(C=0.01, 
          kernel="linear", 
          random_state=0)
svm.fit(X_train, y_train.ravel())
y_pred = svm.predict(X_test)


# In[70]:


cm = confusion_matrix(y_test, y_pred)
disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm.classes_)
disp.plot()
plt.show()


# In[71]:


target_names = ["class '0'", "class '1'", "class '2'", "class '3'"]
print(classification_report(y_test, y_pred, target_names=target_names))


# <div style=" font-size:14px; line_height:160%">
# Let's predict the test data:

# In[72]:


X = df.drop("price_range", axis=1).values
y = df.price_range.values.reshape(-1, 1)
# X, y are now both 2D arrays


# In[73]:


scaler = StandardScaler()

X_scaling = scaler.fit_transform(X)
DF_scaling = scaler.transform(DF.values)

X_scaled = pd.DataFrame(X_scaling, columns=df.drop("price_range", axis=1).columns)
DF_scaled = pd.DataFrame(DF_scaling, columns=DF.columns)


# In[74]:


svm.fit(X_scaled, y.ravel())

predicted_prices_svm = svm.predict(DF_scaled)
predicted_prices_svm = pd.DataFrame(predicted_prices_svm.reshape(-1, 1), columns=["price_range"])

mobile_prices_svm = pd.concat([DF, predicted_prices_svm], axis=1)
mobile_prices_svm


# <div style=" font-size:14px; line_height:160%">
# The accuracy of the Decsion Tree model: 0.8275

# <div style=" font-size:14px; line_height:160%">
# The accuracy of the Random Forest model: 0.8575

# <div style=" font-size:14px; line_height:160%">
# The accuracy of the SVM model: 0.955

# <div style=" font-size:18px; line_height:160%">
# Conclusion:

# <div style=" font-size:14px; line_height:160%">
# As you can see, this dataset was very noisy and that's why models like Decision Tree and Random Forest did not perform well because they are sensitive to noise and overfit on noisy data. On the opposite point, SVM is a noise-resistant algorithm because it works only in terms of support vectors, and for this reason, it performs best on this dataset.

# In[ ]:





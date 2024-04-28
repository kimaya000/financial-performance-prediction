#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 10:45:51 2024

@author: kimayashringarpure
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:54:29 2024

@author: kimayashringarpure
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 19:57:08 2024

@author: kimayashringarpure
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, KFold
train=pd.read_csv("/Users/kimayashringarpure/Downloads/train-5.csv", index_col=False)
test=pd.read_csv("/Users/kimayashringarpure/Downloads/financial-performance-prediction/test.csv", index_col=False)


train.shape
print(train.shape)
test.shape
print(test.shape)
print(test.columns)

nullcheck=train.isnull().sum()
nullcheck=nullcheck.sort_values(ascending=False)



lissst=list(nullcheck.index)
print(lissst)

row_null=train.isnull().sum(axis=1).tolist()

train.shape
print(train.shape)
test.shape
print(test.shape)
print(test.columns)






from sklearn.preprocessing import LabelEncoder
lb=LabelEncoder()
for i in test.columns:
  if test[i].dtype=='object':
    test[i]=lb.fit_transform(test[i])

for i in train.columns:
  if train[i].dtype=='object':
    train[i]=lb.fit_transform(train[i])
    
correlation=train.corr()
correlation_q0=correlation.loc['Q0_TOTAL_ASSETS': 'Q0_EBITDA', : ]
correlation_mean=correlation_q0.mean(axis=0).sort_values(ascending=False)
compare_dataframe=pd.DataFrame([])
hmm=list(correlation_mean.index)
for i in hmm:
    print(i)
    compare_dataframe.loc[i, 'Correlation']=correlation_mean[i]
    compare_dataframe.loc[i, 'Nullvalues']=nullcheck[i]

compare_dataframe=compare_dataframe.sort_values(by='Nullvalues', ascending=False)
print(list(compare_dataframe.index))
nullcheck2=train.isnull().sum()
print(nullcheck2)
train.head(10)

train=train.drop(['trailingPE', 'overallRisk', 'auditRisk',
'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
'Q7_NET_INCOME', 'Q10_COST_OF_REVENUES','Q3_NET_INCOME',
'Q10_REVENUES','Q7_COST_OF_REVENUES', 'Q7_REVENUES',
'Q2_NET_INCOME','Q1_NET_INCOME', 'Q9_COST_OF_REVENUES', 'Q9_REVENUES',
"Q10_fiscal_year_end",'Q9_fiscal_year_end','Q8_fiscal_year_end',
'Q6_fiscal_year_end','Q5_fiscal_year_end','Q4_fiscal_year_end',
'Q1_fiscal_year_end','financialCurrency','Q1_DEPRECIATION_AND_AMORTIZATION',
'Q2_DEPRECIATION_AND_AMORTIZATION','Q4_COST_OF_REVENUES','targetMeanPrice', 
'Q4_COST_OF_REVENUES', 'Q5_COST_OF_REVENUES', 'Q6_COST_OF_REVENUES',
 'Q7_OPERATING_INCOME', 'Q2_fiscal_year_end', 'Q3_fiscal_year_end',
 'recommendationMean', 'recommendationKey'], axis=1)
test=test.drop(['trailingPE', 'overallRisk', 'auditRisk',
'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
'Q7_NET_INCOME', 'Q10_COST_OF_REVENUES','Q3_NET_INCOME',
'Q10_REVENUES','Q7_COST_OF_REVENUES', 'Q7_REVENUES',
'Q2_NET_INCOME','Q1_NET_INCOME', 'Q9_COST_OF_REVENUES', 'Q9_REVENUES',
"Q10_fiscal_year_end",'Q9_fiscal_year_end','Q8_fiscal_year_end',
'Q6_fiscal_year_end','Q5_fiscal_year_end','Q4_fiscal_year_end',
'Q1_fiscal_year_end','financialCurrency','Q1_DEPRECIATION_AND_AMORTIZATION',
'Q2_DEPRECIATION_AND_AMORTIZATION','Q4_COST_OF_REVENUES','targetMeanPrice', 
'Q4_COST_OF_REVENUES', 'Q5_COST_OF_REVENUES', 'Q6_COST_OF_REVENUES',
 'Q7_OPERATING_INCOME', 'Q2_fiscal_year_end', 'Q3_fiscal_year_end', 
 'recommendationMean', 'recommendationKey'], axis=1)
nullcheck=train.isnull().sum()
nullcheck=nullcheck.sort_values(ascending=False)
print(list(nullcheck.index))
#test data cleaning
nullcheck=test.isnull().sum()
nullcheck.sort_values(ascending=False)
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
impute1=KNNImputer(n_neighbors=5)
impute2=SimpleImputer(missing_values=np.nan,strategy="mean")
from missforest.missforest import MissForest
impute3=MissForest()

import numpy as np

# Check for infinity or large values
problematic_cols = []
for col in train.columns:
    if np.any(np.isinf(train[col])) or np.any(np.abs(train[col]) > 1e15):
        problematic_cols.append(col)

# Handle problematic columns
for col in problematic_cols:
    train[col].replace([np.inf, -np.inf], [3.4028235e+38, -3.4028235e+38], inplace=True)
  
# Now you can proceed with imputation
for i in train.columns:
    train[i] = impute2.fit_transform(train[i].values.reshape(-1,1))
    

# Check for infinity or large values
problematic_cols = []
for col in test.columns:
    if np.any(np.isinf(test[col])) or np.any(np.abs(test[col]) > 1e15):
        problematic_cols.append(col)

# Handle problematic columns
for col in problematic_cols:
    test[col].replace([np.inf, -np.inf], [3.4028235e+38, -3.4028235e+38], inplace=True)
     

# Now you can proceed with imputation
for i in test.columns:
    test[i] = impute2.fit_transform(test[i].values.reshape(-1,1))
 

std=train.var().sort_values()
print(list(std.index)) 
corr=train.corr()
high_corr_var=np.where(corr>0.98)
high_corr_var=[(corr.columns[x],corr.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var
dependent='Q0_REVENUES,Q0_COST_OF_REVENUES,Q0_GROSS_PROFIT,Q0_OPERATING_EXPENSES,Q0_EBITDA,Q0_OPERATING_INCOME,Q0_TOTAL_ASSETS,Q0_TOTAL_LIABILITIES,Q0_TOTAL_STOCKHOLDERS_EQUITY'
dependent_list=dependent.split(',')
print(dependent_list)

independent_list=[]
for i in test.columns:
  if(i in dependent_list):
    continue
  else:
    independent_list.append(i)

independent_list

train_mm=train

test_mm=test

X_train = train_mm[independent_list]
X_test=test_mm[independent_list]

Y_train_1=train_mm[['Q0_TOTAL_ASSETS']]
Y_train_2=train_mm[['Q0_TOTAL_LIABILITIES']]
Y_train_3=train_mm[['Q0_TOTAL_STOCKHOLDERS_EQUITY']]
Y_train_4=train_mm[['Q0_GROSS_PROFIT']]
Y_train_5=train_mm[['Q0_COST_OF_REVENUES']]
Y_train_5=train_mm[['Q0_REVENUES']]
Y_train_6=train_mm[['Q0_OPERATING_INCOME']]
Y_train_7=train_mm[['Q0_OPERATING_EXPENSES']]
Y_train_8=train_mm[['Q0_EBITDA']]


import xgboost as xg

xgr=xg.XGBRegressor()

xgr1=xg.XGBRegressor()

predicted_data1 = pd.DataFrame(test['Id'], index=X_test.index)  # Ensure DataFrame has an index

for i in dependent_list:
    xgr.fit(X_train, train_mm[[i]])
    from sklearn.feature_selection import SelectFromModel
    selection = SelectFromModel(xgr, threshold=0.00105, prefit=True)
    selected_dataset = selection.transform(X_train)
    selected_dataset2 = selection.transform(X_test)
    xgr1.fit(selected_dataset, train_mm[i])
    
 
    predicted_data1[i] = xgr1.predict(selected_dataset2)


predicted_data1=predicted_data1.set_index('Id')



predicted_data1.to_csv('/Users/kimayashringarpure/Downloads/financial-performance-prediction/subbmm.csv')



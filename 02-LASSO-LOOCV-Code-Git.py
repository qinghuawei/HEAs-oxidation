# -*- coding: utf-8 -*-
import pandas as pd
from numpy import *
import numpy as np
import math , csv ,sys
import matplotlib.pyplot as plt; plt.style.use('seaborn')
from sklearn.linear_model import Lasso

from sklearn.metrics import r2_score


def R(predict, true):
    y_mean = sum(true) / len(true)
    predict_mean = sum(predict) / len(predict)
    N = []
    D1 = []
    D2 = []
    for i in range(len(true)):
        n = (true[i] - y_mean) * (predict[i] - predict_mean)
        N.append(n)
        d1 = (true[i] - y_mean) ** 2
        D1.append(d1)
        d2 = (predict[i] - predict_mean) ** 2
        D2.append(d2)
    R = sum(N) / pow(sum(D1) * sum(D2), 0.5)
    return R



def RRMSE(predict, true):
    N = []
    for i in range(len(true)):
        e = ((predict[i] - true[i]) / true[i]) ** 2
        N.append(e)
    RRMSE = pow(sum(N) / len(true), 0.5)
    return RRMSE


#Load Data

data = pd.read_csv('00-Data-C.csv')

#print(data,type(data))

Property_Set= list(data)[-1]

Feature_Set_Use = list(data)[1:-1]


Alpha_Value_list_Xlabel=np.arange(-5,5,0.2)
#Alpha_Value_list_Xlabel=[1,0.5]
Alpha_Value_list_Xlabel=list(Alpha_Value_list_Xlabel)
Alpha_Value_list=[]
for Alpha_Value_Xlabel in Alpha_Value_list_Xlabel:
    Alpha_Value_list.append(10**(float(-Alpha_Value_Xlabel)))


Feature_Change_R_Average_Value_CrossValidation=[]
Featrue_Change_RRMSE_Average_Value_CrossValidation=[]
Alpha_Value_Change_Coefficients_Value_list = []

for Alpha_Value in Alpha_Value_list:

    #
    KFold =len(data)
    Entire_sample_numbers = len(data)

    One_Fold_sample_numbers = math.ceil(Entire_sample_numbers / KFold)

    y_test_CrossValidation_collect_list = []
    y_predict_CrossValidation_collect_list = []
    R_Value_CrossValidation_collect_list = []
    RMSE_Value_CrossValidation_collect_list = []

    print('Alpha Value:', Alpha_Value)

    for k in range(KFold):



        Feature_Set_Use=Feature_Set_Use

        Property_Set_Use = Property_Set

        data_Feature = data[Feature_Set_Use]
        data_Target = data[Property_Set]


        if k != KFold - 1:

            bug_fix_number = 0
            if KFold == data_Feature.shape[0]:
                if k == 1:
                    bug_fix_number += 1


            X_train = data_Feature
            y_train = data_Target
            for row in range(k * One_Fold_sample_numbers, (k + 1) * One_Fold_sample_numbers + bug_fix_number):

                X_train = X_train.drop(row)
                y_train = y_train.drop(row)

            X_test = data_Feature.loc[k * One_Fold_sample_numbers:(k + 1) * One_Fold_sample_numbers - 1]
            y_test = data_Target.loc[k * One_Fold_sample_numbers:(k + 1) * One_Fold_sample_numbers - 1]


        if k == KFold - 1:

            bug_fix_number = 0
            if KFold == 1:
                bug_fix_number += data.shape[0] - 1
            X_train = data_Feature
            y_train = data_Target
            for row in range(k * One_Fold_sample_numbers, data_Feature.shape[0] - bug_fix_number):

                X_train = X_train.drop(row)
                y_train = y_train.drop(row)

            X_test = data_Feature.loc[k * One_Fold_sample_numbers:]
            y_test = data_Target.loc[k * One_Fold_sample_numbers:]


        rgr = Lasso(alpha=Alpha_Value, fit_intercept=True, random_state=1).fit(X_train, y_train)

        y_test = np.array(y_test)
        y_predict = rgr.predict(X_test)
        Coefficients = rgr.coef_

        if KFold == 1:
            Alpha_Value_Change_Coefficients_Value_list.append(Coefficients)


        if type(y_predict[0]) is np.float64:
            y_predict_toList_list = []
            for float_value in y_predict:
                float_value_list = []
                float_value_list.append(float_value)
                y_predict_toList_list.append(float_value_list)
            y_predict = y_predict_toList_list

        y_predict_CrossValidation_collect_list.append(y_predict)
        y_test_CrossValidation_collect_list.append(y_test)


    y_predict_CrossValidation_collect_list_csv = []
    y_test_CrossValidation_collect_list_csv = []

    for list in y_predict_CrossValidation_collect_list:
        for element in list:
            y_predict_CrossValidation_collect_list_csv.append(element)  # 使用线性回归/Ridge, element为只有一个元素的列表 , 需要element[0]

    for list in y_test_CrossValidation_collect_list:
        for element in list:
            y_test_CrossValidation_collect_list_csv.append(element)


    R_Value_CrossValidation_collect_list = round( r2_score(y_test_CrossValidation_collect_list_csv, y_predict_CrossValidation_collect_list_csv), 4)
    RRMSE_Value_CrossValidation_collect_list = round(RRMSE(y_test_CrossValidation_collect_list_csv, y_predict_CrossValidation_collect_list_csv), 4)


    with open('Regressor-Sklearn-Python Result Save-CrossValidation.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(y_predict_CrossValidation_collect_list_csv)


    Feature_Change_R_Average_Value_CrossValidation.append(round(mean(R_Value_CrossValidation_collect_list), 4))
    Featrue_Change_RRMSE_Average_Value_CrossValidation.append(round(mean(RMSE_Value_CrossValidation_collect_list), 4))




if KFold==1:

    with open('Regressor-Sklearn-LASSO-1Fold-Aplha_Change_Coeffecient-Python Result Save.csv', 'w') as csvfile:
        writer  = csv.writer(csvfile)
        for list in Alpha_Value_Change_Coefficients_Value_list:
            writer.writerow(list)



with open('LASSO-LOOCV-Alpha_Change_R_Value-Python Result Save.csv', 'w') as csvfile:
    writer  = csv.writer(csvfile)
    for list in Feature_Change_R_Average_Value_CrossValidation:
        list=[list]
        writer.writerow(list)

with open('LASSO-LOOCV-Alpha_Change_RRMSE_Value-Python Result Save.csv.csv', 'w') as csvfile:
    writer  = csv.writer(csvfile)
    for list in Featrue_Change_RRMSE_Average_Value_CrossValidation:
        list = [list]
        writer.writerow(list)


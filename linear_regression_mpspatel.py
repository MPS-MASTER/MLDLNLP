# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 23:46:39 2020

@author: mpspatel
"""

## --------------------------Imports ------------------------------------------

import pandas as pd
import numpy as np
import random
random.seed(10)
import matplotlib.pyplot as plt
import math
#%matplotlib inline

# ----------------------------------------------------------------------------

def equation_line(x,m=11,c=11):
    y = m*x+c
    return y

def logistic_equation(x,m=11,c=11):
    y = m*x+c
    y_new = 1/(1+math.exp(-y))
    return y_new

def absolute_error(original_y, calculated_y):
    """
    1.this is a function that calculates absolute difference and then sums
    2. it to give error value
    2.example :
    arrays1 = [1,2]
    arrays1 = [2,3]
    abs = abs(-1)+abs(1) =2
    """
    abs_err = np.sum(np.absolute(original_y- calculated_y))
    #print("abs_err", abs_err)
    return abs_err

def cross_entropy(y_log, y_org):
    
    # Dot product is matrix multiplication and normal multiplication is elementwise
    
    log_err= np.sum(y_org*math.log(y_log, 10)-(1-y_org)*(log((1-y_log),10)))
    return log_err
    
def sse_error(original_y, calculated_y):
    """
    1.this gives sum of sqaured error and those are pretty big 
    2.this helps take care of outliers because those outliers will give big values
    and our model tries to reduce those value
    3.this shows square of the amount of total error of all data points
    """
    sse = np.sum((original_y- calculated_y)**2)
    return sse

def rmse_error(original_y, calculated_y):
    """
    1.this gives root of mean of sum of sqaured error and those are pretty big 
    2.this helps take care of outliers because those outliers will give big values
    and our model tries to reduce those value
    3. this is error per data point
    """
    #print(len(original_y))
    sse = np.sum(((original_y- calculated_y)**2)/len(original_y))
    return np.sqrt(sse)

def change_in_slope(learning_rate, y_original, y_calculated,input_x ):
    m_change = learning_rate*2*(y_original-y_calculated)*input_x
    return np.sum(m_change)/len(m_change)  ## why i am taking mean of total change
    
def change_in_intercept(learning_rate, y_original, y_calculated,input_x ):
    c_change = learning_rate*2*(y_original-y_calculated)
    return np.sum(c_change)/len(c_change) 

def predicted_values(test_x, slope, intercept):
    val = slope*test_x+intercept
    return val

def scatter_plot_actual_vs_predicted(test_x,actual_y, predicted_y):
    plt.figure(figsize=(10,5))
    plt.scatter(test_x, actual_y, label ="actual output")
    plt.scatter(test_x, predicted_y, label ="predicted output")
    plt.legend()
    plt.xlabel("x value or input")
    plt.ylabel("y value or output")



def linear_reg(train_x, train_y, learning_rate = 0.001):
    """
    train_x, test_x, train_y, test_y 
    
    slop, inter = linear_reg(train_x, train_y)
    
    y_predicted = predicted_values(test_x, slop, inter)
    
    scatter_plot_actual_vs_predicted(test_x, test_y, y_predicted)
    
    rmse_val = rmse_error(test_y, y_predicted)
    
    total_learning = {"slope":slop, "intercept" : inter,"root_mean_square_error":rmse_val}
    
    print("total_learning", total_learning)
      
    """
    previous_loss = np.Inf
    slope = 150
    intercept =150
    previous_slope = 0
    previous_intercept = 0
    no_of_iterations =10000
    errors =[]
    for i in range(no_of_iterations):
        print("iteration no:",i, previous_loss)
    #flag = True
        previous_slope = slope
        previous_intercept =intercept
 #=============================================================================
# =============================================================================
#         train_len = len(train_x)/5
#          
#         for j in range(5):
#             ini = j*train_len
#             endd = (j+1)*train_len
#             train_x_new = train_x.iloc[int(ini) : int(endd)]
# =============================================================================
 #=============================================================================
        y_cal = equation_line(train_x, previous_slope,previous_intercept)
        
        slope = change_in_slope(learning_rate,train_y,y_cal, train_x)
        intercept = change_in_intercept(learning_rate,train_y,y_cal, train_x)
        slope = previous_slope+slope
        intercept = previous_intercept +intercept
        #print(y_cal)
        res_abs = absolute_error(train_y, y_cal)
        res_sse = sse_error(train_y, y_cal)
        res_rmse = rmse_error(train_y, y_cal)
        if(no_of_iterations%10 ==0):
            
            if(previous_loss<res_rmse or previous_loss-res_rmse<0.001):
                return  slope, intercept
            previous_loss = res_rmse
        
        errors.append([i,res_rmse])
        #result_freq = res.value_counts()
        #print("abs_error",res_abs, ", sse_error",res_sse, ", rmse_error",res_rmse)
        #print("slope",slope,"intercept",intercept)
        if(i==no_of_iterations-1):
            return  slope, intercept
        
        
def logestic_reg(train_x, train_y, learning_rate = 0.001):
    """
    train_x, test_x, train_y, test_y 
    
    slop, inter = linear_reg(train_x, train_y)
    
    y_predicted = predicted_values(test_x, slop, inter)
    
    scatter_plot_actual_vs_predicted(test_x, test_y, y_predicted)
    
    rmse_val = rmse_error(test_y, y_predicted)
    
    total_learning = {"slope":slop, "intercept" : inter,"root_mean_square_error":rmse_val}
    
    print("total_learning", total_learning)
      
    """
    previous_loss = np.Inf
    slope = 150
    intercept =150
    previous_slope = 0
    previous_intercept = 0
    no_of_iterations =10000
    errors =[]
    for i in range(no_of_iterations):
        print("iteration no:",i, previous_loss)
    #flag = True
        previous_slope = slope
        previous_intercept =intercept
 #=============================================================================
# =============================================================================
#         train_len = len(train_x)/5
#          
#         for j in range(5):
#             ini = j*train_len
#             endd = (j+1)*train_len
#             train_x_new = train_x.iloc[int(ini) : int(endd)]
# =============================================================================
 #=============================================================================
        y_cal = logistic_equation(train_x, previous_slope,previous_intercept)
        
        slope = change_in_slope(learning_rate,train_y,y_cal, train_x)
        intercept = change_in_intercept(learning_rate,train_y,y_cal, train_x)
        slope = previous_slope+slope
        intercept = previous_intercept +intercept
        #print(y_cal)
        cross_entropy_err = cross_entropy(y_cal, train_y)
        
        
        if(no_of_iterations%10 ==0):
            
            if(previous_loss<cross_entropy_err and previous_loss-cross_entropy_err<0.001):
                return  slope, intercept
            previous_loss = res_rmse
        
        errors.append([i,cross_entropy_err])
        #result_freq = res.value_counts()
        #print("abs_error",res_abs, ", sse_error",res_sse, ", rmse_error",res_rmse)
        #print("slope",slope,"intercept",intercept)
        if(i==no_of_iterations-1):
            return  slope, intercept
        

if __name__ =='__main__':
    
    data = pd.read_csv('datasets//TaxiFareActualsData.csv', sep =',')
    # divide data into train and test
    train_x, test_x, train_y, test_y = data.iloc[0:16,0], data.iloc[16:,0], data.iloc[0:16,1], data.iloc[16:,1]
    
    slop, inter = linear_reg(train_x, train_y)
    
    y_predicted = predicted_values(test_x, slop, inter)
    scatter_plot_actual_vs_predicted(test_x, test_y, y_predicted)
    rmse_val = rmse_error(test_y, y_predicted)
    total_learning = {"slope":slop, "intercept" : inter,"root_mean_square_error":rmse_val}
    print("total_learning", total_learning)

# -*- coding: utf-8 -*-
"""
Created on Thu May 21 02:13:29 2020

@author: mpspa
"""

# --------------Logistic regression code --------------------------------------------------

# 1. Equation of line
# 2. Equation of logistic fnction  ==> Extracted value
# 3. Cost function
# 4. Weigths calculation or theta calculation
# 5. Take care of bias term
# 7. prediction



#  ================ Imports =================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class logestic_regression:
    def __init__(self, learning_rate, initial_weights, input_data, Y):
        self.X = input_data
        self.W = initial_weights
        self.lr = learning_rate
        self.Y  = Y
        
    

    def logic_fun(self,z):
        
        return 1/(1+np.exp(-z))
    
    def cal_z(self, X, W):
        #print(self.X.shape, self.W.shape)
        self.z = np.dot(X, W)  # dot product for matrix multiply
        return self.z
    
    def cost_fun(self,original_y, extrated_y, weight_param, Regularization_coeff):
        term1 = np.dot(-original_y.T,np.log10(extrated_y))
        term2 = np.dot((1-original_y).T,np.log10(1-extrated_y))
        reg = Regularization_coeff*np.dot(weight_param.T,weight_param)
        cf = term1 - term2+reg
        return (1/original_y.shape[0])*np.sum(cf)
    def weight_update(self, iterations,original_y):
        accuracy_dict ={"iteration":[], "accuracy":[]}
        previous_accuracy = 0
        counter =0
        for i in range(iterations):
            #print("Iteration", i)
            z = self.cal_z(self.X, self.W)
            #print(z.shape)
            y_predicted = self.logic_fun(z)
            #print(original_y.shape, y_predicted.shape)
            #print(y_predicted.shape, (y_predicted- original_y).shape)
            self.W = self.W-(1/original_y.shape[0])*self.lr*np.dot(self.X.T,(y_predicted- original_y))
            accuracy_dict["iteration"].append(i)
            accuracy_dict["accuracy"].append(self.cost_fun(original_y, y_predicted, self.W, 10))
            if(previous_accuracy-self.cost_fun(original_y, y_predicted, self.W, 10)<0.000001):
                counter +=1
            else:
                counter -=1
            previous_accuracy = self.cost_fun(original_y, y_predicted, self.W, 10)
            if(counter>=5 and i>1000):
                break
            
            #print(self.W.shape)
            #
            #print(self.cost_fun(original_y, y_predicted, self.W, 10),self.W)
            #break
        return self.W, accuracy_dict
    def calculate_output(self, cal_y):
        result_out = []
        for i in cal_y:
            if i>.5:
                result_out.append(1)
            else:
                result_out.append(0)
        output_arr = np.array(result_out)
        return output_arr
        
    
    
    def plot_decision_boundry(self):
        return True
    
    
#def Log_reg(inp,lear_rate, initial_weight_param):
    

def standerize_input( X):
    from sklearn.preprocessing import StandardScaler
    stdsc = StandardScaler()
    X = stdsc.fit_transform(X)
    return X

def transform_inp(inp, degree):
    from sklearn.preprocessing import PolynomialFeatures
    poly_reg = PolynomialFeatures(degree = degree)
    inp_trans = poly_reg.fit_transform(inp)
    return inp_trans

# =======================Lets Plot--------------------

pos = np.array(inp_data[inp_data.y==1])
neg = np.array(inp_data[inp_data.y==0])

    
 #=============================================================================
def plotData(weights,X, result_out):
     plt.figure(figsize=(10,6))
     plt.plot(pos[:,0], pos[:,1], 'k+', label='Admitted')
     plt.plot(neg[:,0], neg[:,1], 'yo', label='Not Admitted')
     plt.xlabel("Exam1 score")
     plt.ylabel('Exam2 score')
     plt.legend()
#     
#     decision_boundry = -(weights[0]+(weights[1]/40)*X[0])/(weights[2])
#     plt.plot(X[0],decision_boundry)
# =============================================================================
#     x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
#     y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
#     x_grid, y_grid = np.meshgrid(np.arange(x_min, x_max, .1),
#                  np.arange(y_min, y_max, .1))
#     print(x_grid.shape)
#     xx = pd.DataFrame(x_grid.ravel(), columns=["x1"])
#     yy = pd.DataFrame(y_grid.ravel(), columns=["x2"])
#     print(X)
#     z = pd.DataFrame({"intercept" : [1]*xx.shape[0]})
#     z["x1"] = xx
#     z["x2"] = yy
#     p = lri.logic_fun(lri.cal_z(z, weights))
#   
#     out= lri.calculate_output(p)
#     out = np.expand_dims(out, axis = 1)
#     plt.plot(p)
#     print(out.shape)
#     p = p.reshape(x_grid.shape)
#    # out = out.resahpe(x_grid.shape)
#     plt.scatter(inp_data[inp_data["y"] == 0]["x1"], inp_data[inp_data["y"] == 0]["x2"],marker="o")
#     plt.scatter(inp_data[inp_data["y"] == 1]["x1"], inp_data[inp_data["y"] == 1]["x2"],marker="x")
#     #plt.contour(x_grid, y_grid, p, levels = [.5]) #displays only decision boundary
#     plt.contourf(x_grid, y_grid, p, levels = [.5,1], alpha =.4)
#     plt.show()
##     
     
     
     
  
# =============================================================================
     plt.grid()

         
 #=============================================================================
# ========================================================


# ==========================Add bias term to input ============================




# =======================Call instance ================
if __name__ =='__main__':
    
    ## ======== import and standerize input ===============================
    
    inp_data = pd.read_csv("datasets\\classifDat1.txt", header = None)
    inp_data.columns = ["x1", "x2","y"]
    X = inp_data.iloc[:, 0:-1].values   # .values convert to numpy arrays
    Y = inp_data.iloc[:, -1].values
    #plt.scatter(X[0], X[1])
    Y = Y.reshape(Y.shape[0],1)
    print(Y.shape)
    X_std = standerize_input(X)
    X_std = pd.DataFrame(X_std, columns =["x1","x2"])
    print("Shape before tranformation", X_std.shape)
    
    
    # ============ draw input to see the polynomial behaviour and then transform input accordingly ====================
    weights = 0
    result_out = 0
    plotData(weights, X_std, result_out)
    
    X_std = transform_inp(X_std,2)
    print("Shape after Transformation",X_std.shape)
    
    # ==========Analyse data and drop features accordingly ==========================================
    # Since here data is circularly seperated we use squared features
    
    
    
    # ===========Insert bias if not using transform method ========================================
    
    #X_std = X_std
    #X_std = np.insert(X_std, 0, 1, axis =1)
    theta = np.zeros((X_std.shape[1],1))

    lri = logestic_regression(0.0001,theta,X_std, Y)   # Object and initialization
    
    print(theta.shape)
    #z = lri.cal_z()
    #y_cal = lri.logic_fun(z)
    #cost = lri.cost_fun(Y,y_cal,theta, 10000.)
    weights, accuracy_ = lri.weight_update(1000, Y)
    print(weights.shape)
    accuracy_df = pd.DataFrame(accuracy_)
    fig, axs = plt.subplots(2)
    fig.suptitle('Vertically stacked subplots')
    plt.xlabel("no of iterations")
    plt.ylabel("cost")
    axs[0].plot(accuracy_df.iloc[:,0], accuracy_df.iloc[:,1],  )
    #axs[1].plot(x, -y)
    #plt.scatter(accuracy_df.iloc[:,0], accuracy_df.iloc[:,1])
    
    #print(weights, accuracy_)
    lri.plot_decision_boundry()
    
    ## Predict================
    z = lri.cal_z(X_std,weights)
    y_prob = lri.logic_fun(z)
    result_out = lri.calculate_output(y_prob)
    #print(y_prob)
    #plotData(lri,weights, X_std, result_out)
    xxx = result_out.reshape(Y.shape[0],1) ==Y
    accuracy = np.count_nonzero(xxx)/xxx.shape[0]
    print("accuracy ==>", accuracy)
    sns.countplot(xxx.reshape(Y.shape[0]))




    
    
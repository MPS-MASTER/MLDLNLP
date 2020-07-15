# Solving Data Imbalance  Problem using Generative modelling(CONDITIONAL VARIATIONAL AUTOENCODERS)



# Introduction : 
Do you want to achieve more than 90% f1-score, Do you want to have a strong and reliable Deep Learning solution for your fraud detection or anomaly detection. You tried all the techniques but didn't get the desired results. Here I am gonna present you the generative way of solving data imbalance problem. We will generate new data points for minority class and use those data points to fill some of the gaps of imbalance class.
## Problem I am gonna solve :
Most of the time you don't get the good results for highly skewed or highly imbalance dataset. In these cases we use precision and recall as accuracy measure.  Here I have used "creditfraud" data from kaggle  to show the generative way of handling class imbalance. Also compared the results obtained by popular techniques vs generative modeling for handling class imbalance.
## Dataset :
You can download from below kaggle link:


https://www.kaggle.com/mlg-ulb/creditcardfraud
## Previous Best Results obtained by different Techniques :

### Accuracieds taken from different solutions on kaggle

Accuracy on Under sampled data :

Logistic Regression:  0.9798658657817729 

KNears Neighbors:  0.9246195096680248 

Support Vector Classifier:  0.9746783159014857 

Decision Tree Classifier:  0.9173877431007686
### Since Logistic Regression has better accuracy we choose it for further prediction


Logistic regression Precision and recall:



Recall Score: 0.90 

Precision Score: 0.76

 F1 Score: 0.82 

Accuracy Score: 0.81


### Some of them got 93% Recall score but very less precision score

### So best is 93% recall and around 98% validation score

## Now I will Use Generative method to improve accuracies :
--------------------------- Start with coding part
    
    ------------------------------ IMPORTS
```python
import warnings
import numpy as np
from keras.layers import Input, Dense, Lambda
from keras.layers.merge import concatenate as concat
from keras.models import Model
from keras import backend as K
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
#from scipy.misc import imsave
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib
    

```python
import pandas as pd
```

```python
import random
random.seed(150)  # for generating same random no always
```

```python


data = pd.read_csv(r"C:\Users\mpspa\Desktop\Kaggle\creditcard.csv\creditcard.csv")

print("Shape :", data.shape)
print("Columns :",data.columns)
columns = data.columns
data.head(3)
```

    Shape : (284807, 31)
    Columns : Index(['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
           'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
           'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount',
           'Class'],
          dtype='object')
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



#### 3o features and 1 output column(Class)
Lets see data distribution with different class
```python
data.Class.value_counts()
```




    0    284315
    1       492
    Name: Class, dtype: int64



```python
print(" class 0 percentage : ", 100*284315/284807)
print(" class 1 percentage : ", 100*492/284807)
```

     class 0 percentage :  99.827251436938
     class 1 percentage :  0.1727485630620034
    

#### Class 1 is fraud class and its only .17% , Data is higly imbalance or skewed

```python
import seaborn as sns
sns.countplot(data.Class)
Import Image
Image.open(r'/images/Blog_CVAE_CREDIT_FRAUD_files/output_24_1.png')
```




    <matplotlib.axes._subplots.AxesSubplot at 0x15a3200a388>




![png](/images/Blog_CVAE_CREDIT_FRAUD_files/output_24_1.png)


### Prerequisite to know before looking at this code

- Pandas 
- Numpy 
- Neural Networks
- Keras 
- Autoencoders

#### For variational auto encoder you can refer to this video : https://www.youtube.com/watch?v=W4peyiOaEFU

- We will generate data for class 1
- Use that data to make better fraud detection classifier


## I will divide my code in 6 parts

1. Data Preparation
1. Encoder
2. Latent Space
3. Decoder
4. Reconstrution of new fraud data
5. Accuracy calculation and Evaluation of fraud detetor model

1. Data Preparation
#### Since i am generating data for class 1 , i will train my model on class 1 only

```python
data1 = data[data.Class ==1]
```

```python
data1.Class.value_counts()
```




    1    492
    Name: Class, dtype: int64


------------------Normalizing data --------------------


```python
from sklearn.preprocessing import Normalizer
```

```python
ssc = Normalizer()
```

```python
data1.iloc[:, :30] = ssc.fit_transform(data1.iloc[:,:30])
```

```python
data1 = pd.DataFrame(data1)
```

```python
fraud_data = data1
fraud_data = fraud_data.iloc[:450, :]

fraud_x = fraud_data.drop("Class", axis =1)
fraud_y = fraud_data.Class
```

```python
fraud_y.shape
```




    (450,)


--------------- Breaking in train and Test -------------------------
--------------- 400 train and 50 test data ----------------------------


```python
train_x = fraud_x.iloc[:400, :]
train_y =  fraud_y.iloc[:400]
test_x  = fraud_x.iloc[400:450, :]
test_y =  fraud_y.iloc[400: 450]
```

```python
train_y.shape, test_y.shape, train_x.shape, test_x.shape
```




    ((400,), (50,), (400, 30), (50, 30))



2. Encoder
We will make encoder
We sample encoder output as normal distribution and it will be the latent space 

```python
m   = 50 # batch size
n_z = 2   # latent space size

encoder_dim1 = 16 # dim of encoder hidden layer
decoder_dim  = 16 # dim of decoder hidden layer

decoder_out_dim = 30 # dim of decoder output layer

activ = 'relu'
optim = Adam(lr=0.001)

n_x = 30  # Input feature dimention
n_y = 1   # Output dimention( its 1 because of binary output 0 and 1)

n_epoch = 100
```

```python
X     = Input(shape=(n_x,))
label = Input(shape=(n_y,))
```

```python
inputs = concat([X, label])
```

```python
encoder_h = Dense(encoder_dim1, activation=activ)(inputs)

mu      = Dense(n_z, activation='linear')(encoder_h)
l_sigma = Dense(n_z, activation='linear')(encoder_h)
```


3. Latent Space


```python
def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(m, n_z), mean=0., stddev=1.)
    return mu + K.exp(l_sigma / 2) * eps

# Sampling latent space
#z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])
```

```python
# Sampling latent space
z = Lambda(sample_z, output_shape = (n_z, ))([mu, l_sigma])


```
Now we add labels to latest space??
It helps you to give your desirable input and get generated data for that
z is style learned  and label part is data we will give 



```python
# merge latent space with label
zc = concat([z, label])
```
there will be n_z+1 values that will be non zero
and will be passed to decoder DEcoder 



3. Decoder 

```python
decoder_hidden = Dense(decoder_dim, activation=activ)
decoder_out    = Dense(decoder_out_dim, activation='sigmoid')

h_p     = decoder_hidden(zc)
outputs = decoder_out(h_p)
```
---------------Defining the loss


Two types of loss
1. Kl divergence
2. Reconstruction


```python
def vae_loss(y_true, y_pred):

    #recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)

    return recon + kl

def KL_loss(y_true, y_pred):
    return(0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1))

def recon_loss(y_true, y_pred):
    print(y_true)
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
```
----------------- Create CVAE, Encoder, Decoder layers


```python
cvae    = Model([X, label], outputs)
encoder = Model([X, label], mu)

d_in = Input(shape=(n_z+n_y,))
d_h  = decoder_hidden(d_in)
d_out   = decoder_out(d_h)
decoder = Model(d_in, d_out)
```
-------------------- Compile the model


```python
cvae.compile(optimizer=optim, 
             loss=vae_loss, 
             metrics = [KL_loss, recon_loss])
```
-------------------- Fit and train the model


```python
# compile and fit
cvae_hist = cvae.fit([np.array(train_x), np.array(train_y).reshape(len(train_y),1)], 
                     np.array(train_x), 
                     verbose = 1, 
                     batch_size=m, 
                     epochs=n_epoch,
                     validation_data= ([np.array(test_x), np.array(test_y).reshape(len(test_y),1)], 
                     np.array(test_x)),
                     callbacks = [EarlyStopping(patience = 10)])
```
------------------  Plot the loss


```python
# plots - Loss
plt.plot(cvae_hist.history['loss'])
plt.plot(cvae_hist.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper right');
```

5. Reconstrution of new fraud data
    - function to create input vector that will be passed to decoder
    - output of decoder will be the desired generated data

```python
def construct_numvec(digit, z = None):
    out = np.zeros((1, n_z + n_y))
    
    out[:,  n_z] = digit
    #print(out)
    if z is None:
        return(out)
    else:
        for i in range(len(z)):
            out[:,i] = z[i]
        #print('out',out)
        return(out)
```

```python
# Check for working of function

sample_3 = construct_numvec(1)
print(sample_3)
```

    [[0. 0. 1.]]
    

```python
decoder.predict(sample_3).shape
```




    (1, 30)



- Generate 10000 data and store it to generated_data list

```python
generated_data = []
```

```python
dig   = 1
sides = 10000
max_z = 1.5

img_it = 0

for i in range(0, sides):
    z1 = (((i / (sides-1)) * max_z)*2) - max_z
    z_ = [z1]
    for j in range(i, i+1):
        z2 = (((j / (sides-1)) * max_z)*2) - max_z
        z_.append(z2)

        
    vec     = construct_numvec(1, z_)
    decoded = decoder.predict(vec)
    generated_data.append(decoded)
        
```

```python
len(generated_data)
```




    10000



```python
generated_data[1]
```




    array([[0.9821484 , 0.02562311, 0.0210631 , 0.04503274, 0.01104978,
            0.02236631, 0.01006939, 0.01174928, 0.0325062 , 0.01133674,
            0.0383959 , 0.01563377, 0.02919051, 0.01739699, 0.02214013,
            0.02282206, 0.01477762, 0.01290983, 0.00711595, 0.01648735,
            0.01017321, 0.01299593, 0.00997366, 0.00823087, 0.03943525,
            0.04243705, 0.0050112 , 0.0029201 , 0.00621581, 0.01028146]],
          dtype=float32)



```python
type(generated_data)
```




    list



```python
# ------------------- Convert list to arry

generated_data = np.array(generated_data)
```

```python
type(generated_data)

```




    numpy.ndarray



```python
generated_data.shape
```




    (0,)



```python
generated_data = generated_data.reshape(10000,30)
generated_data.shape
```




    (10000, 30)



```python
#gen_data_df =  pd.DataFrame(generated_data)
#gen_data_df.shape
```

- Data generated have 30 features 
- We will add label data column to it

```python
generated_data = np.concatenate((generated_data, np.array([1]*10000).reshape(len(generated_data),1)), axis =1)
```

```python
generated_data.shape
```




    (10000, 31)



- concat generated data to Original data

```python
all_inp = np.concatenate((np.array(data),generated_data), axis = 0)
```

```python
all_inp.shape
```




    (294807, 31)



```python
inp_df = pd.DataFrame(all_inp)
```

```python
inp_df.columns = columns
```

```python
inp_df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Time</th>
      <th>V1</th>
      <th>V2</th>
      <th>V3</th>
      <th>V4</th>
      <th>V5</th>
      <th>V6</th>
      <th>V7</th>
      <th>V8</th>
      <th>V9</th>
      <th>...</th>
      <th>V21</th>
      <th>V22</th>
      <th>V23</th>
      <th>V24</th>
      <th>V25</th>
      <th>V26</th>
      <th>V27</th>
      <th>V28</th>
      <th>Amount</th>
      <th>Class</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.0</td>
      <td>-1.359807</td>
      <td>-0.072781</td>
      <td>2.536347</td>
      <td>1.378155</td>
      <td>-0.338321</td>
      <td>0.462388</td>
      <td>0.239599</td>
      <td>0.098698</td>
      <td>0.363787</td>
      <td>...</td>
      <td>-0.018307</td>
      <td>0.277838</td>
      <td>-0.110474</td>
      <td>0.066928</td>
      <td>0.128539</td>
      <td>-0.189115</td>
      <td>0.133558</td>
      <td>-0.021053</td>
      <td>149.62</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.0</td>
      <td>1.191857</td>
      <td>0.266151</td>
      <td>0.166480</td>
      <td>0.448154</td>
      <td>0.060018</td>
      <td>-0.082361</td>
      <td>-0.078803</td>
      <td>0.085102</td>
      <td>-0.255425</td>
      <td>...</td>
      <td>-0.225775</td>
      <td>-0.638672</td>
      <td>0.101288</td>
      <td>-0.339846</td>
      <td>0.167170</td>
      <td>0.125895</td>
      <td>-0.008983</td>
      <td>0.014724</td>
      <td>2.69</td>
      <td>0.0</td>
    </tr>
    <tr>
      <td>2</td>
      <td>1.0</td>
      <td>-1.358354</td>
      <td>-1.340163</td>
      <td>1.773209</td>
      <td>0.379780</td>
      <td>-0.503198</td>
      <td>1.800499</td>
      <td>0.791461</td>
      <td>0.247676</td>
      <td>-1.514654</td>
      <td>...</td>
      <td>0.247998</td>
      <td>0.771679</td>
      <td>0.909412</td>
      <td>-0.689281</td>
      <td>-0.327642</td>
      <td>-0.139097</td>
      <td>-0.055353</td>
      <td>-0.059752</td>
      <td>378.66</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 31 columns</p>
</div>



- check for the combined input shape

```python
inp_df.Class.value_counts()
```




    0.0    284315
    1.0     10492
    Name: Class, dtype: int64



- Apply Logistic Regression for modelling

```python
from sklearn.linear_model import LogisticRegression
```

```python
clf = LogisticRegression()
```

```python
from sklearn.model_selection import train_test_split
```

```python
train_xx, test_xx, train_y, test_y = train_test_split(inp_df.drop("Class", axis =1),  inp_df.Class)
```

```python
train_xx.shape, test_xx.shape, train_y.shape
```




    ((221105, 30), (73702, 30), (221105,))



```python
clf.fit(train_xx, train_y)
```




    LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                       intercept_scaling=1, l1_ratio=None, max_iter=100,
                       multi_class='auto', n_jobs=None, penalty='l2',
                       random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                       warm_start=False)



```python
pred = clf.predict(test_xx)
```

```python
(pred == test_y).value_counts()
```




    True     73327
    False      375
    Name: Class, dtype: int64



```python
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix,accuracy_score
```

```python

print("accuracy", accuracy_score(test_y, pred))
print("precision", precision_score(test_y, pred))
print("recall", recall_score(test_y, pred))
print("f1", f1_score(test_y, pred))
print("confusion", confusion_matrix(test_y, pred))
```

    accuracy 0.9949119426881224
    precision 0.8887376658727458
    recall 0.9819548872180451
    f1 0.9330237542418289
    confusion [[70715   327]
     [   48  2612]]
    

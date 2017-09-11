
# coding: utf-8

# In[1]:

#import modules
import pandas as pd
import numpy as np
import tensorflow as tf


# In[2]:

#load data
print 'Loading data...'
filename = 'data_me.txt'
data = pd.read_csv(filename, delim_whitespace=True)
#print data


# In[3]:

#delete first 2 columns
del data['FID']
del data['IID']
#print data


# In[4]:

#get feature columns
print 'Extracting features'
column_headers = list(data.columns)

#remove label/target
del column_headers[0]
#print column_headers

feature_columns = []
for column in column_headers:
    feature_columns.append(tf.feature_column.numeric_column(key=column))

#print feature_columns


# In[5]:

def load_data(y_name="PHENOTYPE", train_fraction=0.7, seed=None):
    np.random.seed(seed)
    
    x_train = data.sample(frac=train_fraction, random_state=seed)
    x_test = data.drop(x_train.index)
    
    y_train = x_train.pop(y_name)
    y_test = x_test.pop(y_name)
    
    return (x_train, y_train), (x_test, y_test)


# In[6]:

STEPS = 2000
MODEL_DIR = "./nn_models/irri_model"


# In[7]:

print 'Splitting data for training and testing...'
(x_train, y_train), (x_test, y_test) = load_data()
print x_train, y_train
print x_test, y_test


# In[8]:

input_train = tf.estimator.inputs.pandas_input_fn(
    x=x_train, y=y_train, num_epochs=None, shuffle=True)


# In[9]:

input_test = tf.estimator.inputs.pandas_input_fn(
    x=x_test, y=y_test, shuffle=True)


# In[ ]:

print 'Setting up model...'
model = tf.estimator.DNNRegressor(hidden_units=[200, 200, 150], 
                                  feature_columns=feature_columns,
                                  model_dir=MODEL_DIR,
                                  optimizer=tf.train.AdamOptimizer(
                                                    learning_rate=0.003,
                                                    name="Optimizer"
                                  )
                                 )


# In[ ]:

print 'Training the model...'
model.train(input_fn=input_train, steps=STEPS)


# In[ ]:

print 'Evaluating model'
eval_result = model.evaluate(input_fn=input_test)


# In[ ]:

print 'Calculating loss'
average_loss = eval_result["average_loss"]


# In[ ]:

print("\n" + 80 * "*")
print("\nRMS error for the test set: ${:.0f}".format(average_loss**0.5))

print()
print 'Done.'


# In[ ]:




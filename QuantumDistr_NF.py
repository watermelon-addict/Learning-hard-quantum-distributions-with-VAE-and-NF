# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:43:48 2023

Learning hard quantum distributions using Normalizing Flows 
following similar method as NormalizingFlows2.py https://dmol.pub/dl/flows.html

Here we treat each qubit as a separate dimension for normalzing flows
Have tested for 3 or 8 qubits

How to run: set N = 8 or 3 then F5

input sample size = 5e6

Normalizing flows hyperparameters:
    layers = 5
    epochs = 20
    batch_size = 5000

@author: Ellen Wang
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, kstest
import json

N=8 # 8 or 3 or 12 qubits data

# conver binary array to integer number
def BinStr2Num( Y, N ):
    num = 0
    for i, b in enumerate(Y):
        num += b * 2 **(N-1-i)
    return int(num)

# conver integer number to binary string
def Num2BinStr( num, N ):
    Y = np.zeros(N)
    for i in range(N):
        b = num % 2
        num = num//2
        Y[N-1-i] = b
    return Y

distr = "Quantum" # options are "Triangular" or "Quantum"
# distr = "Triangular" # triangular does not work here

np.random.seed(123456789)
tf.random.set_seed(123456789)

input_size = 5000000 # input sample size

tfd = tfp.distributions
tfb = tfp.bijectors

data_ub = 2**N

qsim_data = True # quantum data from QSIM tutorial https://pennylane.ai/qml/demos/qsim_beyond_classical/
# QSIM data we use 8 or 12 qubits
# paper quantum data we use 3 or 8 qubits

# datay is our target distribution either Triangular or The Quantum distr
if( distr == "Quantum"):
    if (not qsim_data ): # quantum data from the paper
        if( N == 3):
            saved_file = "Quantum_VAE_Binary_Data_3bit.json"
        else:
            saved_file = "Quantum_VAE_Binary_Data.json"
    else: # quantum data from QSIM tutorial https://pennylane.ai/qml/demos/qsim_beyond_classical/
        # saved_file = "QSIM_Binary_Data_{}bit.json".format(N)
        saved_file = "QSIM_Binary_Data_{}bit_SI.json".format(N)

    norm_factor = 2**N

    with open( saved_file, "r") as openfile:
        jso = json.load(openfile)
    y_out = jso["data"]
        
    datay = np.array(y_out[ 0 : input_size ])
    data = [ Num2BinStr(x, N) for x in datay ]
    data = np.asarray(data)

plt.hist( datay, bins=2**N, range = (0,2**N-1) )
plt.title( "Quantum distribution data - %d qubits" % N )
plt.show()

# show a sample scatter plot for the first and second qubits, this has no real effect for training
plt.plot( data[:,0], data[:,1], ".", alpha =0.2) 
plt.title( "Quantum distribution data - %d qubits: scatter plot of 1st vs. 2nd qubits" % N )
plt.show()

tfd = tfp.distributions
tfb = tfp.bijectors

zdist = tfd.MultivariateNormalDiag(loc=[0.0] * N)

num_layers = 5
my_bijects = []
# loop over desired bijectors and put into list
perm = np.linspace(N-1, 0, N) # e.g. [2,1,0] for 3 qubits
perm = [ int(x) for x in perm ]
for i in range(num_layers):
    # Syntax to make a MAF
    anet = tfb.AutoregressiveNetwork(
        # params=2, hidden_units=[128,128], activation="relu"
        params=2, hidden_units=[128,128], activation="sigmoid"
    )
    ab = tfb.MaskedAutoregressiveFlow(anet)
    # Add bijector to list
    my_bijects.append(ab)
    # Can't permuate 1 dimension
    permute = tfb.Permute(perm)
    my_bijects.append(permute)
# put all bijectors into one "chain bijector" that looks like one
big_bijector = tfb.Chain(my_bijects)
# make transformed dist
td = tfd.TransformedDistribution(zdist, bijector=big_bijector) 

# declare the feature dimen sion
x = tf.keras.Input(shape=(N,), dtype=tf.float32)
# create a "placeholder" function that will be model output
log_prob = td.log_prob(x)
# use input (feature) and output (log prob) to make model
model = tf.keras.Model(x, log_prob)

# define a loss
def neg_loglik(yhat, log_prob): 
    # losses always take in label, prediction  
    # in keras. We do not have labels,
    # but we still need to accept the arg
    # to comply with Keras format
    return -log_prob


# now we prepare model for training
model.compile(optimizer=tf.optimizers.Adam(1e-3), loss=neg_loglik)

model.summary()

plot_model(model, to_file="nf2.png", show_shapes=False, 
           show_layer_names=False, show_layer_activations=False, 
           show_trainable=False)

result = model.fit(x=data, y=np.zeros(input_size), epochs=20, batch_size = 50000, verbose=1)
# training not stable, got nan very often?

plt.plot(result.history["loss"])
plt.title( "Normalizing flow loss - %d qubits" % N )
plt.show()

'''
# probability density distribution
zpointsx = np.linspace(0, 1, 200)
( z1, z2, z3 ) = np.meshgrid(zpointsx, zpointsx, zpointsx)
zgrid = np.concatenate((z1.reshape(-1, 1), z2.reshape(-1, 1), z3.reshape(-1,1)), axis=1)

p = np.exp(td.log_prob(zgrid))
fig = plt.figure()
plt.imshow(
    p.reshape(z1.shape), aspect="equal", origin="lower", extent=[0,1,0,1]
)
plt.show()
'''

# now sample from trained distr
# scatter plot
zsamples = td.sample( input_size/10 ) # don't need to generate too large size of data; here we pick 1/5 of original data size
zss = []
for i in range(N):
    zs1 = np.asarray(zsamples[:,i] )
    zs1b = [ 1 if x>0.5 else 0 for x in zs1]
    zss.append( zs1b )
plt.plot( zss[0], zss[1], ".", alpha =0.2) 
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title( "Generated data after training - Quantum Distribution %d qubits: scatter plot of 1st vs. 2nd qubits" % N)
plt.show()

# convert output sample binary string to number to show the distribtion in 1D histogram and compare with original
zs = zss[0]
for i in range(1,N):
    zs = np.column_stack((zs, zss[i]))
# zs = np.stack((zs1b, zs2b, zs3b ), axis = -1)
zsy = [ BinStr2Num( x, N ) for x in zs ]
plt.hist( zsy, bins=2**N, range = (0,2**N-1) )
plt.title( "Generated data after training - Quantum Distribution %d qubits" % N )
plt.show()

# compare distance of output samples with original distribution
wd = wasserstein_distance(datay, zsy)
print( "Wasserstein distance between x_test[0] and NF output is: ", wd)
ksr = kstest(datay, zsy)
print( "KS test between x_test[0] and NF output is: ", ksr)

# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 22:43:48 2023

Train Normalizing Flows network to learn 2 types of quantum distributions:
First distribution is proposed by B. Fefferman and C. Chris Umans, "The poer of quantum Fourier sampling", arXiv:1507.05592, 2015.
Second is the quantum distribution generated from a qsim circuit.
Additionally, we also test on 2 types of simple classical distributions: Guassian and Poisson.

The research framework and results are described in paper:
"Learning hard quantum distributions with generative neural networks", by E. Wang, et al.

This script trains an NF to learn the hard quantum distributions, generate outputs and compare distribtion with original.
The NF setup follows the implementation in https://dmol.pub/dl/flows.html

This program provides 2 options of underlying probability distribution data:
    1. The hard quantum distribution by Fefferman and Umans.
    2. The hard quantum distribution from a qsim circuit.

How to run this code?
Set N = 3 or 8 for Fefferman; 8 or 12 for qsim, then F5
    1. Quantum Fefferman & Umans: set distr = "Quantum_F_U"
    2. Quantum qsim: set distr = "Quantum_qsim"
    Normalizing flows hyperparameters:
        layers = 5
        epoches = 20
        batch_size = 5000

The program generates the following outputs:
    1. A sample input distribution histogram i.e. Gaussian, Poisson or Quantum
    2. Training loss history
    3. A sample output distribution histogram from test input
    4. Scipy.stats Wasserstein distance and kstest results of the above outputs from 3 & 4


The project is inspired by Rocchetto, Andrea et al.,
"learning hard quantum distributions with variational autoencoders", NPJ Quantum Information 2018.

@author: Ellen Wang
"""

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from keras.utils import plot_model
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance, kstest
import json

N = 3 # number of qubits in quantum distribution
# Fefferman & Umans quantum data we use 3 or 8 qubits
# QSIM data we use 8 or 12 qubits

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

# Which distribution do we train VAE?
distr = "Quantum_F_U" # options are "Quantum_F_U", or "Quantum_qsim"
distr = "Quantum_qsim"

np.random.seed(123456789)
tf.random.set_seed(123456789)

input_size = 5000000 # input sample size

tfd = tfp.distributions
tfb = tfp.bijectors

data_ub = 2**N

if( distr == "Quantum_F_U"):
    saved_file = "Quantum_Distr_Fefferman_{}bit.json".format(N)
elif( distr == "Quantum_qsim"):
    saved_file = "Quantum_Distr_qsim_{}bit.json".format(N)
else:
    saved_file = ""

norm_factor = 2**N

with open( saved_file, "r") as openfile:
    jso = json.load(openfile)
y_out = jso["data"]
    
# datay is our target distribution
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

# start with standard normal distribution
zdist = tfd.MultivariateNormalDiag(loc=[0.0] * N)

# build the NF network
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

# Calculate numeric difference between the training data and VAE-generated data distribution
# 2 metrics are calculated: Wasserstein distance, and KS test
wd = wasserstein_distance(datay, zsy)
print( "Wasserstein distance between x_test[0] and NF output is: ", wd)
ksr = kstest(datay, zsy)
print( "KS test between x_test[0] and NF output is: ", ksr)

# -*- coding: utf-8 -*-
"""
Created on Tue Sep  12 21:10:02 2023

Train VAE network to learn 2 types of quantum distributions:
First distribution is proposed by B. Fefferman and C. Chris Umans, "The poer of quantum Fourier sampling", arXiv:1507.05592, 2015.
Second is the quantum distribution generated from a qsim circuit.
Additionally, we also test on 2 types of simple classical distributions: Guassian and Poisson.


The research framework and results are described in paper:
"Learning hard quantum distributions with generative neural networks", by E. Wang, et al.

This script trains a variational autoencoder to learn the hard quantum distributions, generate outputs and compare distribtion with original.

This program provides 4 options of underlying probability distribution data:
    1. Gaussian continuous distribution, for simple testing purpose.
    2. Poisson discrete distribution, for simple testing purpose.
    3. The hard quantum distribution by Fefferman and Umans.
    4. The hard quantum distribution from a qsim circuit.

How to run this code?
Set N = 3 or 8 for Fefferman; 8 or 12 for qsim, then F5
    1. Gaussian: set distr = "Gaussian"
    2. Poisson: set distr = "Poisson"
    3. Quantum Fefferman & Umans: set distr = "Quantum_F_U"
    4. Quantum qsim: set distr = "Quantum_qsim"

The program generates the following outputs:
    1. A sample input distribution histogram i.e. Gaussian, Poisson or Quantum
    2. Training loss history
    3. A sample output distribution histogram from test input
    4. Nine output distribution histograms from choosing Gaussian samples of the latent space
    5. Scipy.stats Wasserstein distance and kstest results of the above outputs from 3 & 4


The project is inspired by Rocchetto, Andrea et al.,
"learning hard quantum distributions with variational autoencoders", NPJ Quantum Information 2018.

@author: Ellen Wang
"""

import numpy as np
import cmath 
import math
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Dense, Input, LeakyReLU
from keras.layers import Flatten, Lambda, Reshape
from keras import backend as K
from keras.losses import binary_crossentropy, mean_squared_error, categorical_crossentropy
from keras.utils import plot_model
import tensorflow as tf
from scipy.stats import wasserstein_distance, kstest

# below are the flags on how to run this code:

# Which distribution do we train VAE?
distr = "Quantum_F_U" # options are "Gaussian", "Poisson", "Quantum_F_U", or "Quantum_qsim"
distr = "Quantum_qsim"
distr = "Gaussian"
distr = "Poisson"

N = 3 # number of qubits in quantum distribution
# Fefferman & Umans quantum data we use 3 or 8 qubits
# QSIM data we use 8 or 12 qubits

# below are functions used in this project

# A function to display distribution by random guassian sampling of the latent space
# latent space sample mean and stdev can be set up to (0,1) or the average of z_mean and z_log_var from training sets
def display_new_distr( num_distr = (3,3), latent_dim = 8, norm_factor = 256,
                      zz_mean = np.zeros(8), zz_log_var = np.ones(8), train_x0 = np.zeros(1) ):
    new_lats = np.empty( ( num_distr[0] * num_distr[1], latent_dim ) )
    counter = 0
    for i in range( num_distr[0] ):
        for j in range( num_distr[1]):
            new_lat = []
            for k in range( latent_dim ):
                new_lat.append(np.random.normal(zz_mean[k], np.exp(zz_log_var[k])))
            new_lats[counter,:] = new_lat
            counter += 1
    new_distr = decoder.predict(new_lats)*norm_factor
    fig, axes = plt.subplots(num_distr[0], num_distr[1], sharex=True,
                             sharey=True, figsize=(20, 20))
    counter = 0
    for i in range( num_distr[0] ):
        for j in range( num_distr[1]):
            axes[i][j].hist(new_distr[counter], bins=2**N, range = (0,2**N-1) )
            wd = wasserstein_distance(train_x0, new_distr[counter])
            print( "Wasserstein distance between first training set and random-latent-sample VAE output ", counter, " is: ", wd)
            ksr = kstest(train_x0, new_distr[counter])
            print( "KS test between first training set and random-latent-sample VAE output ", counter, " is: ", ksr)
            counter += 1
    plt.suptitle( "VAE output with Gaussian random sample of latent vectors")
    plt.show()


if( distr == "Quantum_F_U"):
    saved_file = "Quantum_Distr_Fefferman_{}bit.json".format(N)
elif( distr == "Quantum_qsim"):
    saved_file = "Quantum_Distr_qsim_{}bit.json".format(N)
else:
    saved_file = ""

# Now we have the quantum binary distributions
# Now build a VAE
if( distr == "Quantum_F_U" or distr == "Quantum_qsim" ):
    norm_factor = 2**N
    with open( saved_file, "r") as openfile:
        jso = json.load(openfile)
    y_out = jso["data"]
        
    if( N > 10 ): # if more than 10 qubits, needs more data for each sample
        sample_sz = 1000
        input_dim = 5000    
    else:
        sample_sz = 1000
        input_dim = 5000    

    y_out = np.array(y_out[ 0 : sample_sz*input_dim ]) # 5 million samples

    y_out = y_out.reshape((sample_sz, input_dim))
    y_out_used = []
    for x in y_out:
        y_out_used.append(np.sort(x)) 
    y_out_used = np.array(y_out_used)        
elif( distr == "Gaussian" or distr == "Poisson" ):
    y_out_used = []
    y_bins = []
    norm_factor= 20 
    
    # Gaussian mean and stdev
    mu = 10 # can be anything
    sigma = 2
    
    # poisson lambda
    lam = 5  # set to any number >=1
    sample_sz = 1000
    input_dim = 5000
    for _ in range(sample_sz):    
        if( distr == "Gaussian"):
            y = np.random.normal( mu, sigma, input_dim )
        else: # Poisson
            y = np.random.poisson( lam, input_dim )

        y = np.sort(y) # sort this so it can be compared with final decoder output which will also be sorted
        y_out_used.append(y)     
    y_out_used = np.array(y_out_used)                
else:
    raise Exception('Unrecognized distribution')
        
plt.hist( y_out_used.reshape((sample_sz*input_dim,)), bins=2**max(N,8), range = (0,2**N-1) ) # display the first training set with 5000 samples
plt.title( "Full training sample")
plt.show()

plt.hist( y_out_used[0,:], bins=2**N, range = (0,2**N-1) ) # display the first training set with 5000 samples
plt.title( "First training sample")
plt.show()
y_out_used = y_out_used / norm_factor

# split to train and validation
x_train, x_test = train_test_split(
    y_out_used, test_size=0.2, random_state=42
)
print( x_train.shape, x_test.shape )

input_dim = x_train.shape[1]
latent_dim = 8 # can be different numbers. For Gaussian or Poisson distributions, even 1 latent dimension is sufficient

# encoder: 2 dense layers
enc_input = Input(shape=(input_dim)) 
enc_dense1 = Dense(units=500, activation="relu")(enc_input) # hidden layer dimension chosen to be 500, can be different number
enc_activ1 = LeakyReLU()(enc_dense1)
enc_dense2 = Dense(units=latent_dim)(enc_activ1)
enc_output = LeakyReLU()(enc_dense2)

shape = K.int_shape( enc_output )
print( shape )

# add variational guassian for latent vectors
def sampling(args):
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z_mean = Dense(latent_dim, name='z_mean')( enc_output )
print(K.int_shape(z_mean))

z_log_var = Dense(latent_dim, name='z_log_var')( enc_output )
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# encoder with variatinoal gaussian
encoder = Model(enc_input, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

# Sort decoder output so it can be compared with input which is sorted
# Otherwise how do we compare these 2 distributions? 
# Next mini-projecti might be to build a Wasserstein distance custom loss function using keras.backend
def sortVec(x):    
    sorted_x = tf.sort(x, axis=1, direction='ASCENDING', name=None)
    #sorted_x = sorted_x.reshape(-1,1)
    return(sorted_x)    

# decoder whcih is just reverse of encoder
latent_input = Input(shape=(latent_dim,), name='z_sampling') 
dec_dense1 = Dense(units=500, activation="relu")(latent_input)
dec_activ1 = LeakyReLU()(dec_dense1)
dec_dense2 = Dense(units=input_dim, activation='sigmoid')(dec_activ1)
dec_output = LeakyReLU()(dec_dense2)
dec_output = Lambda(sortVec)(dec_output) # critical layer to sort the decoder output
decoder = Model(latent_input, dec_output)    
decoder.summary() 

# VAE: put encoder and decoder together
encoded_str = encoder(enc_input)
print( "z_mean shape is ", encoded_str[0].shape)    
# print( encoded_str[0]) # 0 is z_mean
# print( encoded_str[1]) # 1 is z_log_var
# print( encoded_str[2]) # 2 is z
dec_output = decoder(encoded_str[2])
vae = Model(enc_input, dec_output, name='vae')
vae.summary()

plot_model(vae, to_file="vae.png", show_shapes=True, 
           show_layer_names=True, show_layer_activations=True, 
           show_trainable=False)

# VAE loss = reconstruction loss between input and decoded, plus KL difference of gaussian
# reconst_loss = mean_squared_error(K.flatten(enc_input), K.flatten(dec_output))
reconst_loss = binary_crossentropy(K.flatten(enc_input), K.flatten(dec_output))
reconst_loss *= input_dim

kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -2
vae_loss = K.mean(reconst_loss + kl_loss)

vae.add_loss(vae_loss)
vae.compile(optimizer='rmsprop')
vae.summary()

history = vae.fit(x_train,epochs=100,batch_size=100,shuffle=True,validation_data=(x_test,None))

#training history plot
plt.plot(history.history['loss'] )
plt.plot( history.history['val_loss'] )
plt.ylim(0, 10000)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel( "epoch")
plt.show()

# check latent space dimension
encoded = encoder(x_train)
print( "latent space shape of one training set is ", K.int_shape(encoded[0] )) 

# get the average of each z_mean and z_log_var, which will be used for later VAE sample generation
zz_mean =[]
zz_log_var = []
for i in range( latent_dim ):
    zz_mean.append( np.average(encoded[0][:,i]))
    zz_log_var.append( np.average(encoded[1][:,i]))

# generate output binary strings from x_test
z_mean, z_log_var, z = encoder.predict(x_test)
decoded_vec = decoder.predict(z_mean) * norm_factor

ss = []
for s in decoded_vec:
    ss = np.concatenate((ss, s))
    
plt.hist( ss, bins=2**N, range = (0,2**N-1) )
plt.title( "VAE output with test input")
plt.show()

# Calculate numeric difference between the training data and VAE-generated data distribution
# 2 metrics are calculated: Wasserstein distance, and KS test
wd = wasserstein_distance(x_test[i,:] * norm_factor, s)
print( "Wasserstein distance between x_test[0] and VAE output is: ", wd)
ksr = kstest(x_test[i,:] * norm_factor, s)
print( "KS test between x_test[0] and VAE output is: ", ksr)

# Displaying latent space distribution, histogram for 1-d latent space; scatter for 2-d latent space
# For 8-dimensional latent space this won't work
# # 1-dimenstional latent space
# plt.hist(encoded[:,0], bins=100, range = (0,2**N-1) )
# plt.show()
# # # 2-dimenstional latent space
# plt.figure(figsize=(12,12))
# plt.scatter(encoded[:,0], encoded[:,1], s=2, cmap='hsv')
# plt.colorbar()
# plt.grid()
# plt.show()

# # Displaying several newly generated distributions
print( "\n\nSample generated from random vectors in latent space")
display_new_distr( num_distr = (3,3), latent_dim = latent_dim, norm_factor = norm_factor,
                  zz_mean = zz_mean, zz_log_var = zz_log_var, train_x0 = x_train[0,:] * norm_factor )


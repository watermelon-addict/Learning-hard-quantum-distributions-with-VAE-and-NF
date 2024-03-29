# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:09:32 2023 

learning hard quantum distributions with variational autoencoders, NPJ Quantum Information 2018
Rocchetto, Andrea et al.

Generate a distribution of quantum binary strings Y
Use variational autoencoder to generate outputs and compare distribtion with original

This program give 3 options of underlying probability distribution data:
    1. Gaussian continuous distribution
    2. Poisson discrete distribution
    3. The complex binary quantum distribution as in the paper 

How to run this code?
    1. Gaussian: set distr = "Gaussian" and VAE = True
    2. Poisson: set distr = "Poisson" and VAE = True
    3. Quantum: set distr = "Quantum" and there are 2 steps:
        3.a) First, set generate_quantum_data = True at the beginning and run it once only to generate 
            binary training data which will be saved to a json file
        3.b) Then VAE = True for VAE training

The program generates the following outputs:
    1. A sample input distribution histogram i.e. Gaussian, Poisson or Quantum
    2. Training loss history
    3. A sample output distribution histogram from test input
    4. Nine output distribution histograms from choosing Gaussian samples of the latent space
    5. Scipy.stats Wasserstein distance and kstest results of the above outputs from 3 & 4

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
distr = "Quantum" # options are "Gaussian", "Poisson" or "Quantum"
# distr = "Gaussian" # options are "Gaussian", "Poisson" or "Quantum"
# distr = "Poisson" # options are "Gaussian", "Poisson" or "Quantum"

# For quantum distribution, it takes a long time to generate data so let's run the generation and save data first
# Set this flag to True at the beginning and run it once only to generate binary training data
# This flag is not used if trainging on Gaussian and Poission, for which the generation is done on the fly before training
generate_quantum_data = False

# set this flag to false when you only want to generate quantum data; otherwise set this to true
VAE = True # set this flag to True when doing VAE training
qsim_data = False # quantum data from QSIM tutorial https://pennylane.ai/qml/demos/qsim_beyond_classical/
# QSIM data we use 8 or 12 qubits
# paper quantum data we use 3 or 8 qubits

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


# below are functions to generate quantum binary distribution        

num_prev = 0 # start y converted to number between 0 and 2^N-1, this is a random choice
pr_prev = 0 # used for metropolis initialization

def h(n, k):
    return ((n & (1 << (k - 1))) >> (k - 1))

# conver binary array to integer number
def BinStr2Num( Y, N ):
    num = 0
    for i, b in enumerate(Y):
        num += b * 2 **(N-1-i)
    return int(num)

def prob(Z, n, L, N):
    n_fac = math.factorial(n)
    product = 1.
    q = 0    
    for z in range( n_fac ):
        for x in Z:
            bit = Z.index(x) + 1
            product *= x ** h(z, bit)            
        q += product
    
    # probility of y = |Q|^2 / (L^N * N!)
    pr = (q.real**2 + q.imag**2)/(L**N * n_fac)
    return pr

# we generate a random string Y consisted of 0 and 1
# probabily of this string is determined by a bespoke formula Prob(Q)
def gen_y_distr(L, N, n):
    global pr_prev
    global num_prev
    rng = np.random.default_rng()
    Y = rng.integers(L, size=N)
    
    Z = []
    w = cmath.exp(2*math.pi*(1j)/L)
    for i in Y:
        Z.append(w**i)        
    pr = prob(Z,n,L,N)
    
    # run a simple metropolis algorithm to decide whether to keep Y or not, this generates the desired distribution
    # Accept Y if:
    # 1. the probabity of Y is higher than current state probability, or,
    # 2. if the probabilyt of Y is lower than current state, accept at chance of the proability ratio
    if( pr_prev == 0 ):
        pr_ratio = 1.
    else:
        pr_ratio = pr / pr_prev
    
    if( pr_ratio >= 1. or rng.random() < pr_ratio ):
        num = BinStr2Num(Y, N)
        pr_prev = pr
        num_prev = num
        # y_out holds the binary strings, it's converted to numbers between 0 and 2^N-1
        y_out.append( num )
    else:
        y_out.append( num_prev )
        
    return(Y)


N = 8 # number of binary digits in quantum distribution
if (not qsim_data ): # quantum data from the paper
    if( N == 3):
        saved_file = "Quantum_VAE_Binary_Data_3bit.json"
    else:
        saved_file = "Quantum_VAE_Binary_Data.json"
else: # quantum data from QSIM tutorial https://pennylane.ai/qml/demos/qsim_beyond_classical/
    # saved_file = "QSIM_Binary_Data_{}bit.json".format(N)
    saved_file = "QSIM_Binary_Data_{}bit_SI.json".format(N)

L = 2 
n = 5

# generate binary data via the Q distribution
if( generate_quantum_data ): 
    y_out = []
    y_bins = []
    
    # Set metropolis starting point
    rng = np.random.default_rng()
    Y = rng.integers(L, size=N)
    Z = []
    w = cmath.exp(2*math.pi*(1j)/L)
    for i in Y:
        Z.append(w**i)      
    p_prev = prob(Z, n, L, N)
    num_prev = BinStr2Num(Y, N)
    
    for _ in range(int(5e6)): # generate 5,000,000 samples
        # generate strings of Y,
        # each Y is from 0 to L-1, so for binary Y, we set L=2
        
        # N is the length of Y string, set to 256 in this example
        # n determins the terms in the summation for Q, set to 5 in this example 5! = 120
        gen_y_distr(L, N, n)
        
    tot_sz = len(y_out)
    print( "Number of binary string samples: %d" % tot_sz  )
    # Number of binary string samples: 58700
    
    # show histogram of Y distribution
    plt.hist(y_out, bins=2**N, range = (0,2**N-1) )
    plt.show()
    
    # y_out = [ x.tolist() for x in y_out ]
    # y_out = y_out.tolist()
    with open( saved_file, "w") as outfile:
        data = {"data": y_out}
        json.dump(data, outfile, indent = 4 )

# Now we have the quantum binary distributions
# Now build a VAE
if( VAE ):    
    if( distr == "Quantum"):
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
    
    # VAE
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
    

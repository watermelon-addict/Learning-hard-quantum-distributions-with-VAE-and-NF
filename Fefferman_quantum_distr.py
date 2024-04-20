# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 20:09:32 2023

Generate distributions by quantum Fourier sampling as proposed by Fefferman and Umans in 
B. Fefferman and C. Chris Umans, "The Power of Quantum Fourier Sampling," arXiv:1507.05592, 2015. 

How to run this code?
    Set generate_quantum_data = True  
    Set distr = "Quantum" and there
    Then run to generate binary training data which will be saved to "Quantum_VAE_Binary_Data_{}bit.json"
    Usually do not run multiple times to avoid overwriting the previous saved .json file
    This also plot the distribution histogram

The research framework and results are described in paper:
"Learning hard quantum distributions with generative neural networks", by E. Wang, et al.

@author: Ellen Wang
"""

import numpy as np
import cmath 
import math
import matplotlib.pyplot as plt
import json

distr = "Quantum" 
generate_quantum_data = True

N = 3 # number of qubits
saved_file = "Quantum_Distr_Fefferman_{}bit.json".format(N)

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


# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:11:15 2018

@author: lsalaun

© 2016 - 2020 Nokia
Licensed under Creative Commons Attribution Non Commercial 4.0 International
SPDX-License-Identifier: CC-BY-NC-4.0

"""

import numpy as np
import generate_gains
import channel_allocation_JSPA as ca_jspa

K = 5 # Number of users
L = 1 # Number of subcarriers
# Radius of the cell
Radius = 1000
# Min distance between user and BS = 35 m
rmin = 35

#np.random.seed(1)
G = generate_gains.generateGains(K,L,Radius,rmin)

# B : total bandwidth = 5 MHz
B = 5*10**6
W = np.ones(L)*B/L

# Noise value per Hertz = -174 dBm/Hz
Noise_Hz = 10**((-174-30)/10)
N = np.ones(K*L)*Noise_Hz*B/L
N_G = ca_jspa.computeN_G(G,N)

# Decoding order
pi,pi_inv = ca_jspa.computePi(G,N)

# Random weights
w = np.random.rand(K)

# Considered subcarrier and active users
lsub = 0
ksub = np.arange(K)

# Power budget
Pmax = 1

print('pi =',pi)
print('w =',w)

print('\n----------------------- TEST SCPC on a single subcarrier with 5 users -----------------------')

ca_jspa.SCPC_barplot(K,Pmax,N_G[lsub],lsub,ksub,pi,pi_inv,w,W)

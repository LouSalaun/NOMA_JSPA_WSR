# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 14:11:15 2018

@author: lsalaun
"""

import numpy as np
import matplotlib.pyplot as plt
import generate_gains
import channel_allocation_JSPA as ca_jspa

K = 5 # Number of users
L = 1 # Number of subcarriers
Multiplex = 3 # Number of users per subcarrier
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

# Considered subcarrier
lsub = 0

# Power budget
Pmax = 1

print('pi =',pi)
print('w =',w)
print('G =\n',G)

print('\n----------------------- TEST SCUS and i-SCUS on a single subcarrier with at most 3 active users out of 5  -----------------------')
espilon = 1e-3
X = np.arange(espilon,Pmax,espilon)
Y_SCUS = np.zeros(X.shape)
Y_SpeedUp_SCUS = np.zeros(X.shape)

# speedUp_SCUS initialization
speedUp_SCUS = ca_jspa.SpeedUpSCUS(K,Multiplex,Pmax,N_G[lsub],lsub,pi,pi_inv,w,W)

for index in range(len(X)):
    # SCUS computation
    x_f2l = ca_jspa.SCUS_first2last(K,Multiplex,X[index],N_G[lsub],lsub,pi,w,W)
    p_f2l = ca_jspa.X2P_sc(x_f2l,pi,K,lsub)
    Y_SCUS[index] = np.sum(ca_jspa.computeWSR_sc(K,lsub,N_G[lsub],pi,pi_inv,w,W,p_f2l))
    
    # speedUp_SCUS computation
    x,p,wsr = speedUp_SCUS.speedUp_SCUS(X[index])
    Y_SpeedUp_SCUS[index] = np.sum(wsr)

plt.figure(0)
plt.plot(X,Y_SCUS,'-',label='SCUS')
plt.plot(X,Y_SpeedUp_SCUS,'o',markevery=0.05,label='i-SCUS (speed up)')
plt.xlabel('Power budget (W)')
plt.ylabel('WSR (bit/s)')
plt.legend()
plt.show()
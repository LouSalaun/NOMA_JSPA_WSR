# -*- coding: utf-8 -*-
"""
Created on Mon Sep  5 14:56:35 2016

@author: lsalaun
"""

import math
import numpy as np

def generateGains(K,L,R,rmin):
    
    # Generate the position of K users in 1 cell
    # Uniform in a circle without r<rmin
    BUdistance = rmin + math.sqrt(R**2-rmin**2)*np.sqrt(np.random.rand(K))   
#    print(BUdistance)

    # Compute the link gain including Rayleigh fading, path loss and shadowing
    rayleigh = np.random.randn(K,L) + 1j*np.random.randn(K,L)   # fast fading

    path_loss = -(128.1+37.6*np.log10(BUdistance/1000))   # path loss model : BUdistance/1000 in km
    path_loss = np.power(10,(path_loss/10))               # dB to scalar
    
    shadowing = -10*np.random.randn(K,L)          # lognormal distributed with SD 10
    shadowing = np.power(10,(shadowing/10))       # dB to scalar
    
    BUlinkgain = np.array([[ path_loss[k] * np.power(np.absolute(rayleigh[k][l]),2) * shadowing[k][l] for l in range(L)] for k in range(K)])
    
    '''
    print('BUdistance',BUdistance)   
    print('rayleigh',rayleigh) 
    print('path_loss',path_loss) 
    print('shadowing',shadowing) 
    print('BUlinkgain',BUlinkgain) 
    '''
    
    return BUlinkgain
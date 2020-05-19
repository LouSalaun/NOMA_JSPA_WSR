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

K = 2 # Number of users
L = 10 # Number of subcarriers
Multiplex = 1 # Number of users per subcarrier
# Radius of the cell
Radius = 1000
# Min distance between user and BS = 35 m
rmin = 35

np.random.seed(0)  # Seed
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

# Power budget
Pmax = 1
# Power budget per subcarrier
PmaxN = Pmax*np.ones(L)

print('pi =',pi)
print('w =',w)
print('G =\n',G)

print('\n----------------------- opt_JSPA with delta = 0.01  -----------------------')
# Power discretization step
delta = 0.01

power, wsr, countSCUS, countDP = ca_jspa.opt_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,delta)

print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of SCUS evaluations performed:',countSCUS)
print('Number of iterations (for loops) in the dynamic programming:',countDP)

print('\n----------------------- opt_JSPA with delta = 0.001  -----------------------')
# Power discretization step
delta = 0.001

power, wsr, countSCUS, countDP = ca_jspa.opt_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,delta)

print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of SCUS evaluations performed:',countSCUS)
print('Number of iterations (for loops) in the dynamic programming:',countDP)

print('\n----------------------- epsilon_JSPA with espilon = 0.1  -----------------------')
# Approx epsilon
epsilon = 0.1

power, wsr, countSCUS, countNormWSR, countDP = ca_jspa.epsilon_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,epsilon)

print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of SCUS evaluations performed:',countSCUS)
print('Number of computed WSR values:',countNormWSR)
print('Number of iterations (for loops) in the dynamic programming:',countDP)

print('\n----------------------- epsilon_JSPA with espilon = 0.01  -----------------------')
# Approx epsilon
epsilon = 0.01

power, wsr, countSCUS, countNormWSR, countDP = ca_jspa.epsilon_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,epsilon)

print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of SCUS evaluations performed:',countSCUS)
print('Number of computed WSR values:',countNormWSR)
print('Number of iterations (for loops) in the dynamic programming:',countDP)

print('\n----------------------- Grad_JSPA with delta = 0.001, with rounding_step = 0.01 -----------------------')
# Error tolerance at termination
delta = 0.001

power, wsr, countIters = ca_jspa.SpeedUp_Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=0.01)
print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of gradient iterations:',countIters)

# The following Grad_JSPA gives the same result but with larger running time since it uses the basic SCUS instead of SpeedUpSCUS
#power, wsr, countIters = ca_jspa.Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=0.01)
#print()
#print('WSR performance:',np.sum(wsr))
#print('Power allocated to each subcarrier:',power)
#print('Number of gradient iterations:',countIters)

print('\n----------------------- Grad_JSPA with delta = 0.001, without rounding -----------------------')
# Error tolerance at termination
delta = 0.001

power, wsr, countIters = ca_jspa.SpeedUp_Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=None)
print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)
print('Number of gradient iterations:',countIters)

# The following Grad_JSPA gives the same result but with larger running time since it uses the basic SCUS instead of SpeedUpSCUS
#power, wsr, countIters = ca_jspa.Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=None)
#print()
#print('WSR performance:',np.sum(wsr))
#print('Power allocated to each subcarrier:',power)
#print('Number of gradient iterations:',countIters)

print('\n----------------------- eqPow_JSPA -----------------------')

power, wsr = ca_jspa.eqPow_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W)
print()
print('WSR performance:',np.sum(wsr))
print('Power allocated to each subcarrier:',power)

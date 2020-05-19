# -*- coding: utf-8 -*-
"""
Created on Tue May  9 15:39:50 2017

@author: lsalaun

© 2016 - 2020 Nokia
Licensed under Creative Commons Attribution Non Commercial 4.0 International
SPDX-License-Identifier: CC-BY-NC-4.0

"""

import math
import numpy as np
import matplotlib.pyplot as plt

# K : nb users
# L : nb subbands
# Multiplex : max number of users multiplexed on each subcarrier
# Pmax : total max power
# PmaxN : max power per subcarrier
# G : link gains (not normalized) -> array (K,L)
# N : noises

# pi and pi_inv dim = [L]x[K]
# pi[n] : ranking -> user's index
# pi_inv[n] : user's index -> its ranking

# Compute pi and pi_inv
def computePi(G,N):
    # Find K and L
    K = len(G)
    L = len(G[0])
    pi = np.zeros((L,K),dtype=np.int8)
    pi_inv = np.zeros((L,K),dtype=np.int8)
    
    # G_N gain over noise ratio array, dim = [L]x[K]
    G_N = np.array([ [G[i][n]/N[i*L+n] for i in range(K)] for n in range(L) ])
    
    # Compute pi 
    pi = np.argsort(G_N, axis=1).astype(np.int8)
    pi_inv = np.zeros((L,K), dtype=np.int8)
    for k in range(K):
        for n in range(L):
            pi_inv[n][pi[n][k]] = k

    return pi,pi_inv

# compute the N over G array of dim = [L]x[K]
def computeN_G(G,N):
    # Find K and L
    K = len(G)
    L = len(G[0])
    N_G = np.array([[N[i*L+n]/G[i][n] for i in range(K)] for n in range(L) ])
    
    return N_G

# compute the N over G array of dim = [K*L]
def computeN_Gvect(G,N):
    # Find K and L
    K = len(G)
    L = len(G[0])
    N_G = np.zeros((K*L))
    for k in range(K):
        for l in range(L):
            N_G[L*k+l] = N[L*k+l]/G[k][l]
    
    return N_G

# Compute normalized interference plus noise (valid for downlink only)
def NormIplusN_DL(K,L,N_G,pi,pi_inv,p):
    # Normalized interference plus noise vector
    v = np.zeros(K*L)
    for n in range(L):
        for k in range(K):
            # Add normalized noise
            v[k*L+n] += N_G[n][k]
            rank = pi_inv[n][k]
            # Add the normalized interference from user pi[n][rank2] such that rank2 > rank  
            for rank2 in range(rank+1,K):
                k2 = pi[n][rank2]
                v[k*L+n] += p[k2*L+n]        
    return v

# Compute weighted data rates vector
def computeWeightedR(K,L,N_G,pi,pi_inv,w,W,p):
    v = NormIplusN_DL(K,L,N_G,pi,pi_inv,p)
    C = np.zeros((K,L))
    for u in range(K):
        for f in range(L):
            C[u][f] = W[f]*w[u]*math.log2(1+p[L*u+f]/v[L*u+f])
    return C

# Compute WSR for a single subcarrier 
def computeWSR_sc(K,lsub,N_G_l,pi,pi_inv,w,W,p):
    # Normalized interference plus noise vector
    v = np.zeros(K)
    for k in range(K):
        # Add normalized noise
        v[k] += N_G_l[k]
        rank = pi_inv[lsub][k]
        # Add the normalized interference from user pi[lsub][rank2] such that rank2 > rank  
        for rank2 in range(rank+1,K):
            k2 = pi[lsub][rank2]
            v[k] += p[k2]   
    
    # compute the WSR of subcarrier lsub   
    C = W[lsub]*w*np.log2(1+p/v)
#    print('p',p)
#    print('v',v)
#    print('N_G_l',N_G_l)
#    print(W[lsub]*w*np.log2(1+p/v))
#    print('C',C)
    return C

# Compute WSR for a single subcarrier given:
# x: a power vector x of type X (transformed) with only active users. 
# The active users indexes are given by "users"
# users: the list of active users 
# This is used in SpeedUpSCUS for faster computation of the wsr (time complexity from K to M)
def computeWSR_sc_X(users,lsub,N_G_l,pi,pi_inv,w,W,x):
    # Number of active users
    M = len(users)
    # Normalized interference plus noise vector
    v = np.zeros(len(users))
    for index_user in range(M-1):
        rank = users[index_user]
        k = pi[lsub][rank]
        # Add normalized noise
        v[index_user] += N_G_l[k]
        # Add the normalized interference from user with index i > index_user and i < M-1
        for i in range(index_user+1,M-1):
            v[index_user] +=  x[i] - x[i+1]
        # Last element 
        v[index_user] +=  x[M-1] 
        
    # Last decoded user (best user)
    rank = users[M-1]
    k = pi[lsub][rank]
    # Add normalized noise
    v[M-1] += N_G_l[k]
    
    # compute the WSR of subcarrier lsub 
    C = np.zeros(M)
    for index_user in range(M-1):
        rank = users[index_user]
        k = pi[lsub][rank]
        C[index_user] = W[lsub]*w[k]*np.log2(1+(x[index_user]-x[index_user+1])/v[index_user])
    rank = users[M-1]
    k = pi[lsub][rank]
    C[M-1] = W[lsub]*w[k]*np.log2(1+x[M-1]/v[M-1])

#    print('v',v)
#    print('C',C)
    return C

########################################### SCPC and SCUS algorithms ###########################################

# Convert x (of size [L]x[K]) to p of size [K*L]
def X2P(x,pi,K,L):
    p = np.zeros(K*L)
    for l in range(L):
        # p_{pi^n(i)} = x_i^l - x_{i+1}^l
        for i in range(K-1):
            p[pi[l,i]*L+l] = x[l][i] - x[l][i+1]
        # Last element 
        p[pi[l,K-1]*L+l] = x[l][K-1]
    return p

def X2P_sc(x,pi,K,lsub):
    p = np.zeros(K)
    # p_{pi^n(i)} = x_i^l - x_{i+1}^l
    for i in range(K-1):
        p[pi[lsub,i]] = x[i] - x[i+1]
    # Last element 
    p[pi[lsub,K-1]] = x[K-1]
    return p

# Convert short active X list to X
def activeX2X_sc(active_x,active_users,pi,K,lsub):
    x = np.zeros(K)
    for index in range(len(active_x)-1,-1,-1):
        x[0:active_users[index]+1] = active_x[index]
    return x


# Maximize f_i,j^l
# l : considered subcarrier
# P is the max allocated power budget    
# w : weights
# W : subcarrier's bandwidth
# N_G_l : N/G only restrict to subcarrier l (column l)
def MaxF(j,i,l,N_G_l,pi,P,w):
    a = pi[l][i]
    b = pi[l][j-1]
    if j == 0 or w[a] >= w[b]:
        return P
    else:
        return max(0,min((w[b]*N_G_l[a]-w[a]*N_G_l[b])/(w[a]-w[b]),P))
    
# Evaluate f_i,j^l
def F(x,j,i,l,N_G_l,pi,w,W):
    a = pi[l][i]
    b = pi[l][j-1]
    if j == 0:
        return W[l]*w[a]*math.log2(x+N_G_l[a])
    elif j > i:
        return 0
    else:
        return W[l]*(w[a]*math.log2(x+N_G_l[a])-w[b]*math.log2(x+N_G_l[b]))

# Evaluate the derivative of f_i,j^l with respect to x 
def F_deriv(x,j,i,l,N_G_l,pi,w,W):
    a = pi[l][i]
    b = pi[l][j-1]
    if j == 0:
        return W[l]/math.log(2)*w[a]/(x+N_G_l[a])
    elif j > i:
        return 0
    else:
        return   W[l]/math.log(2)*(w[a]/(x+N_G_l[a])-w[b]/(x+N_G_l[b]))  

# SCPC : single-carrier power control
# lsub : current subcarrier
# ksub : subset of users (i.e., active users)
# P : current subcarrier max power allocation
def SCPC(K,P,N_G_l,lsub,ksub,pi,pi_inv,w,W):   
    # Convert ksub to their ranking (decoding order) vector x_index
    x_index = np.sort(pi_inv[lsub,ksub])
    # Vector of x of length K
    x = np.zeros(K)
    for index in range(len(x_index)):
        i = x_index[index]
        x_star = MaxF(i,i,lsub,N_G_l,pi,P,w)
        index2 = index-1
        j = x_index[index2]
        
        while index2 >= 0 and x[j]<x_star:
            x_star = MaxF(j,i,lsub,N_G_l,pi,P,w)
            index2 = index2-1
            j = x_index[index2]
            
        x[j+1:i+1] = x_star

    return x

# SCPC with bar plot for testing or demoing
def SCPC_barplot(K,P,N_G_l,lsub,ksub,pi,pi_inv,w,W):   
    # Convert ksub to their ranking (decoding order) vector x_index
    x_index = np.sort(pi_inv[lsub,ksub])
    # Vector of x of length K
    x = np.zeros(K)
    for index in range(len(x_index)):
        i = x_index[index]
        x_star = MaxF(i,i,lsub,N_G_l,pi,P,w)
        index2 = index-1
        j = x_index[index2]
        
        print('\n---------------- Step i =',i,'----------------')
        
        # TEST
        x[i] = x_star
        plt.bar(np.arange(K), x, color="blue")
        plt.show()
        print('x =',x)
        # TEST
        
        while index2 >= 0 and x[j]<x_star:
            print('\n---------------- Step i =',i,'----------------')
            print('----------- Backtracking step j =',j,'-----------')
            x_star = MaxF(j,i,lsub,N_G_l,pi,P,w)
            index2 = index2-1
            j = x_index[index2]
            
            # TEST
            x[j+1:i+1] = x_star
            plt.bar(np.arange(K), x, color="blue")
            plt.show()
            print('x =',x)
            # TEST
            
        x[j+1:i+1] = x_star

    return x

# F^n derivative with respect to P
# x : result of SCPC (or SCUS) with input P for subcarrier l
# l : current subcarrier
def Fn_derivP(K,P,x,l,N_G_l,pi,w,W):   
    j = 0
    i = 0
    while i+1<K and x[i+1] == P:
        i += 1

    return F_deriv(P,j,i,l,N_G_l,pi,w,W)

# SCUS_first2last : single-carrier user selection (dynamic programming) from the Infocom paper
# Going from first decoded user (index 1) to last decoded user (index K)
# lsub : current subcarrier
# P : current subcarrier max power allocation
def SCUS_first2last(K,Multiplex,P,N_G_l,lsub,pi,w,W):
    
    # Initialize table V, T
    V = np.zeros((Multiplex,K,K))
    X = np.zeros((Multiplex,K,K))
    T = np.zeros((Multiplex,K,K,3),dtype=np.int8)
    for i in range(K):
        V[0,0,i] = F(P,0,i,lsub,N_G_l,pi,w,W) + F(0,i+1,K-1,lsub,N_G_l,pi,w,W)
        X[0,0,i] = P
        T[0,0,i,:] = -1*np.ones(3,dtype=np.int8)        
    for j in range(1,K):
        for i in range(j,K):
            to_P = F(P,0,i,lsub,N_G_l,pi,w,W) + F(0,i+1,K-1,lsub,N_G_l,pi,w,W)
            to_0 = V[0,j-1,j-1]
            if to_P >= to_0:
                V[0,j,i] = to_P
                X[0,j,i] = P
                T[0,j,i,:] = np.array([0,0,i],dtype=np.int8)
            else:
                V[0,j,i] = to_0
                X[0,j,i] = 0
                T[0,j,i,:] = np.array([0,j-1,j-1],dtype=np.int8)                
    for m in range(1,Multiplex):
        for i in range(K):
            V[m,0,i] = V[0,0,i]
            X[m,0,i] = X[0,0,i]
            T[m,0,i,:] = -1*np.ones(3,dtype=np.int8)
    
    # Iterates
    for j in range(1,K):
        for m in range(1,Multiplex):
            # Update
            # v1 : cas ou j~i est indep. de 0,..,j-1
            # v2 : cas ou j~i est intégré à 0,..,j-1
            for i in range(j,K):
                x = MaxF(j,i,lsub,N_G_l,pi,P,w)
                v0 = V[m,j-1,j-1] 
                if x >= X[m-1,j-1,j-1] or x<=0: 
                    v1 = -math.inf
                else:
                    v1 = V[m-1,j-1,j-1] + F(x,j,i,lsub,N_G_l,pi,w,W) - F(0,j,i,lsub,N_G_l,pi,w,W)
                v2 = V[m,j-1,i]
                # Update
                if v1 > v2 and v1 > v0:
                    V[m,j,i] = v1
                    X[m,j,i] = x
                    T[m,j,i,:] = np.array([m-1,j-1,j-1],dtype=np.int8)
                elif v2 > v0:
                    V[m,j,i] = v2
                    X[m,j,i] = X[m,j-1,i]
                    T[m,j,i,:] = np.array([m,j-1,i],dtype=np.int8)
                else:
                    V[m,j,i] = v0
                    X[m,j,i] = 0
                    T[m,j,i,:] = np.array([m,j-1,j-1],dtype=np.int8)                    
    
    # Retrieve the best allocation from V, X, T
    x = np.zeros(K)
    m = Multiplex-1
    i = 0
    valmax = -math.inf
    for l in range(K):
        if valmax <= V[m,l,l]:
            valmax = V[m,l,l]
            i = l
    j = i
    while True:
        x[j:i+1] = X[m,j,i]
        if T[m,j,i,0] == -1:
            break
        newT = T[m,j,i,:]
        m = newT[0]
        j = newT[1]
        i = newT[2]
    
#    print(V[Multiplex-1,:,:])
    
    return x

# SCUS_last2first : single-carrier user selection (dynamic programming) from the TSP paper
# Going from last decoded user (index K) to first decoded user (index 1)
# lsub : current subcarrier
# P : current subcarrier max power allocation
def SCUS_last2first(K,Multiplex,P,N_G_l,lsub,pi,w,W):
        
    # Initialize table V, T
    V = np.zeros((Multiplex+1,K,K))
    X = np.zeros((Multiplex+1,K,K))
    T = np.zeros((Multiplex+1,K,K,3),dtype=np.int8)
    
    # Initialization ----------------------------------------------------------------------
    # Multiplex = 0 means no active users (every x = 0)
    for i in range(K-1,-1,-1):
        for j in range(i,-1,-1):
            V[0,j,i] = F(0,j,K-1,lsub,N_G_l,pi,w,W)
            X[0,j,i] = 0
            T[0,j,i,:] = -1*np.ones(3,dtype=np.int8)    
    for m in range(1,Multiplex+1):
        for j in range(K-1,-1,-1):
            x_opt = MaxF(j,K-1,lsub,N_G_l,pi,P,w)
            V[m,j,K-1] = F(x_opt,j,K-1,lsub,N_G_l,pi,w,W)
            X[m,j,K-1] = x_opt
            T[m,j,K-1,:] = -1*np.ones(3,dtype=np.int8)
            
    # Iteration ----------------------------------------------------------------------
    for i in range(K-2,-1,-1):
        for m in range(1,Multiplex+1):
            for j in range(i,-1,-1):
                x_opt = MaxF(j,i,lsub,N_G_l,pi,P,w)
                V_if_i_active = F(x_opt,j,i,lsub,N_G_l,pi,w,W)+V[m-1,i+1,i+1]
                V_if_i_inactive = V[m,j,i+1]
                if V_if_i_active > V_if_i_inactive and x_opt > X[m-1,i+1,i+1]: 
                    V[m,j,i] = V_if_i_active
                    X[m,j,i] = x_opt
                    T[m,j,i,:] = np.array([m-1,i+1,i+1],dtype=np.int8)
                else:
                    V[m,j,i] = V_if_i_inactive
                    X[m,j,i] = X[m,j,i+1]
                    T[m,j,i,:] = np.array([m,j,i+1],dtype=np.int8)     
            
    # Retrieve the best allocation from V, X, T
    x = np.zeros(K)
    m = Multiplex
    i = 0
    j = 0
    while True:
        x[j:i+1] = X[m,j,i]
        if T[m,j,i,0] == -1:
            break
        newT = T[m,j,i,:]
        m = newT[0]
        j = newT[1]
        i = newT[2]
            
    return x

########################################### Grad_JSPA and eqPow_JSPA algorithm ###########################################

# This class contains a speed up version of SCUS when it is called multiple times
# on the same subcarrier with differente power budget P
class SpeedUpSCUS:
    # When the object is instantiate, SCUS is called once for budget Pmax (max possible power budget on this subcarrier)
    # then at most K users selections (power vector x) are kept in self.users_selections to later speed up the single-carrier user selectiona and power allocation
    # In later use, SCUS can be simplified and computed in O(MK) (see method speedUp_SCUS)
    def __init__(self,K,Multiplex,Pmax,N_G_l,lsub,pi,pi_inv,w,W):
        self.K = K
        self.lsub = lsub
        self.Multiplex = Multiplex
        self.Pmax = Pmax
        self.N_G_l = N_G_l
        self.pi = pi
        self.pi_inv = pi_inv
        self.w = w
        self.W = W

        #---------------- Below is equivalent to SCUS_last2first(K,Multiplex,Pmax,N_G_l,lsub,pi,w,W) ----------------
        # Initialize table V, T
        V = np.zeros((Multiplex+1,K,K))
        X = np.zeros((Multiplex+1,K,K))
        T = np.zeros((Multiplex+1,K,K,3),dtype=np.int8)
    
        # Initialization ----------------------------------------------------------------------
        # Multiplex = 0 means no active users (every x = 0)
        for i in range(K-1,-1,-1):
            for j in range(i,-1,-1):
                V[0,j,i] = F(0,j,K-1,lsub,N_G_l,pi,w,W)
                X[0,j,i] = 0
                T[0,j,i,:] = -1*np.ones(3,dtype=np.int8)    
        for m in range(1,Multiplex+1):
            for j in range(K-1,-1,-1):
                x_opt = MaxF(j,K-1,lsub,N_G_l,pi,Pmax,w)
                V[m,j,K-1] = F(x_opt,j,K-1,lsub,N_G_l,pi,w,W)
                X[m,j,K-1] = x_opt
                T[m,j,K-1,:] = -1*np.ones(3,dtype=np.int8)
                
        # Iteration ----------------------------------------------------------------------
        for i in range(K-2,-1,-1):
            for m in range(1,Multiplex+1):
                for j in range(i,-1,-1):
                    x_opt = MaxF(j,i,lsub,N_G_l,pi,Pmax,w)
                    V_if_i_active = F(x_opt,j,i,lsub,N_G_l,pi,w,W)+V[m-1,i+1,i+1]
                    V_if_i_inactive = V[m,j,i+1]
                    if V_if_i_active > V_if_i_inactive and x_opt > X[m-1,i+1,i+1]: 
                        V[m,j,i] = V_if_i_active
                        X[m,j,i] = x_opt
                        T[m,j,i,:] = np.array([m-1,i+1,i+1],dtype=np.int8)
                    else:
                        V[m,j,i] = V_if_i_inactive
                        X[m,j,i] = X[m,j,i+1]
                        T[m,j,i,:] = np.array([m,j,i+1],dtype=np.int8)     
                
        # Retrieve the best allocation for V[Multiplex,j,i] with i = 0..K-1 and j = 0
        # Store the corresponding power vector x in list_power_vectors
        list_power_vectors = np.zeros((K,K))
        for i_init in range(K):
            m = Multiplex
            i = i_init
            j = 0
            while True:
                list_power_vectors[i_init,j:i+1] = X[m,j,i]
                if T[m,j,i,0] == -1:
                    break
                newT = T[m,j,i,:]
                m = newT[0]
                j = newT[1]
                i = newT[2]
            
        # Only keep the power vectors that are differents
        # i.e. remove redundant power vectors in the list
        list_power_vectors = np.unique(list_power_vectors, axis=0)
        # Format list_power_vectors by keeping only the active users' power
        # and store their index in list_active_users
        self.list_power_vectors = [None] * len(list_power_vectors)
        self.list_active_users = [None] * len(list_power_vectors)
        for index in range(len(list_power_vectors)):
#            print(index)
            power_vector = list_power_vectors[index]
            active_users = []
            active_power_vector = []
            user_pt = 0
            while user_pt < len(power_vector):
                while user_pt+1 < len(power_vector) and power_vector[user_pt+1]==power_vector[user_pt]:
                    user_pt += 1
                active_users.append(user_pt)
                active_power_vector.append(power_vector[user_pt])
                user_pt += 1
            # Remove last power if it's equal to zero
            if active_power_vector[-1] == 0:
                active_users.pop()
                active_power_vector.pop()
                
            self.list_power_vectors[index] = np.array(active_power_vector)
            self.list_active_users[index] = np.array(active_users,dtype=np.int)
        
        #---------------- End of SCUS_last2first ----------------

    # speedUp_SCUS, uses list_power_vectors to simplify the calculation
    # P is the current power budget, P should be less than or equal to Pmax
    def speedUp_SCUS(self, P):
        # Find which power allocation is best
        best_x = np.minimum(self.list_power_vectors[0],P*np.ones(self.list_power_vectors[0].shape))
        best_wsr = np.sum(computeWSR_sc_X(self.list_active_users[0],self.lsub,self.N_G_l,self.pi,self.pi_inv,self.w,self.W,best_x))
        best_index = 0
        for i in range(1,len(self.list_power_vectors)):
            curr_x = np.minimum(self.list_power_vectors[i],P*np.ones(self.list_power_vectors[i].shape))
            curr_wsr = np.sum(computeWSR_sc_X(self.list_active_users[i],self.lsub,self.N_G_l,self.pi,self.pi_inv,self.w,self.W,curr_x))
            if curr_wsr > best_wsr:
                best_x = curr_x
                best_wsr = curr_wsr
                best_index = i

        # get the full description X (with all users: actives and inactives)
#        print('ground truth SCUS',SCUS_last2first(self.K,self.Multiplex,P,self.N_G_l,self.lsub,self.pi,self.w,self.W))
#        print('best short X',best_x)
#        print('avtice users',self.list_active_users[best_index])
        best_x = activeX2X_sc(best_x,self.list_active_users[best_index],self.pi,self.K,self.lsub)
#        print('best_x',best_x)
        best_p = X2P_sc(best_x,self.pi,self.K,self.lsub)
#        print('best_p',best_p)
        best_wsr = computeWSR_sc(self.K,self.lsub,self.N_G_l,self.pi,self.pi_inv,self.w,self.W,best_p)
        
        return best_x, best_p, best_wsr
    
    # Same as speedUp_SCUS, but only returns the wsr (the scalar, not the vector)
    def speedUp_SCUS_onlyWSR(self, P):
        # Find which power allocation is best
        best_x = np.minimum(self.list_power_vectors[0],P*np.ones(self.list_power_vectors[0].shape))
        best_wsr = np.sum(computeWSR_sc_X(self.list_active_users[0],self.lsub,self.N_G_l,self.pi,self.pi_inv,self.w,self.W,best_x))
        for i in range(1,len(self.list_power_vectors)):
            curr_x = np.minimum(self.list_power_vectors[i],P*np.ones(self.list_power_vectors[i].shape))
            curr_wsr = np.sum(computeWSR_sc_X(self.list_active_users[i],self.lsub,self.N_G_l,self.pi,self.pi_inv,self.w,self.W,curr_x))
            if curr_wsr > best_wsr:
                best_x = curr_x
                best_wsr = curr_wsr

        return best_wsr

########################################### Grad_JSPA and eqPow_JSPA algorithm ###########################################

# Function to minimize by Grad_JSPA
# Pvect : vector of each subcarriers' power budget to be optimized
# Output : function value, gradient
def fun_first2last(Pvect,K,L,Multiplex,N_G,pi,pi_inv,w,W):
    x = np.zeros((L,K))
    for lsub in range(L):
        x[lsub] = SCUS_first2last(K,Multiplex,Pvect[lsub],N_G[lsub],lsub,pi,w,W)
    p = X2P(x,pi,K,L)
    
    # Compute the gradient 
    grad = np.zeros(L)
    for l in range(L):
#        print(Fn_derivP(K,Pvect[l],x[l],l,N_G[l],pi,w,W))
        grad[l] = Fn_derivP(K,Pvect[l],x[l],l,N_G[l],pi,w,W)
        
#    print('Pvect',Pvect)
#    print('funval',-np.sum(computeWeightedR(K,L,N_G,pi,pi_inv,w,W,p)))
#    for l in range(L):
#        print([p[k*L+l] for k in range(K)])    
#    print('grad',-grad)
        
    return -np.sum(computeWeightedR(K,L,N_G,pi,pi_inv,w,W,p)), -grad

# Function to minimize by Grad_JSPA
# Uses the speed up SpeedUpSCUS procedure
# suSCUS : List of initialized SpeedUpSCUS objects, one for each subcarrier
# Pvect : vector of each subcarriers' power budget to be optimized
# Output : function value, gradient
def fun_last2first(Pvect,suSCUS,K,L,Multiplex,N_G,pi,pi_inv,w,W):
    x = np.zeros((L,K))
    wsr = np.zeros((L,K))
    for lsub in range(L):
        x[lsub], _, wsr[lsub] = suSCUS[lsub].speedUp_SCUS(Pvect[lsub])
    
    # Compute the gradient 
    grad = np.zeros(L)
    for l in range(L):
#        print(Fn_derivP(K,Pvect[l],x[l],l,N_G[l],pi,w,W))
        grad[l] = Fn_derivP(K,Pvect[l],x[l],l,N_G[l],pi,w,W)
        
#    print('Pvect',Pvect)
#    print('funval',-np.sum(computeWeightedR(K,L,N_G,pi,pi_inv,w,W,p)))
#    for l in range(L):
#        print([p[k*L+l] for k in range(K)])    
#    print('grad',-grad)
        
    return -np.sum(wsr), -grad

# Function to get the weighted sum rate given Pvect
# this function calls SCUS_first2last
def F_Pvect_first2last(Pvect,K,L,Multiplex,N_G,pi,pi_inv,w,W):
    x = np.zeros((L,K))
    for lsub in range(L):
        x[lsub] = SCUS_first2last(K,Multiplex,Pvect[lsub],N_G[lsub],lsub,pi,w,W)
#    print(x)
    p = X2P(x,pi,K,L)
#    print(p)
#    for l in range(L):
#        print([p[k*L+l] for k in range(K)])
    
    return computeWeightedR(K,L,N_G,pi,pi_inv,w,W,p)

# Projection of P on the simplex with sum P <= Pmax
def proj(P,Pmax):
    return P*Pmax/np.sum(P)

# Grad_JSPA : multi-carrier power control -> projected gradient on top of SCUS
# l : current subcarrier
# Pmax : total power budget
# PmaxN : max power per subcarrier
# delta : variable precision at termination
# max_iters : maximum number of iterations (100 by default)
# rounding_step : if not None, then the best solution found is rounded to the nearest multiple of 
#                 rounding_step. i.e., the per subcarrier power allocation is discretized to 
#                 take value of the form l*rounding_step, where l=0..floor(Pmax/rounding_step) 
# Output: best power allocation per subcarrier found, corresponding objective function value (WSR), 
#         number of iterations performed by the gradient 
def Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,max_iters=100,rounding_step=None):
    # Initial value
    P0 = Pmax/L*np.ones(L)

    # Projected gradient descent 
    previous_step_size = math.inf
    max_iters = 100 # maximum number of iterations
    iters = 0 #iteration counter
    # Step length: well chosen fixed step length alpha performs well
    # Backtracking line search can also be implemented but requires more computations
    alpha = 0.1/np.max(W)/np.max(w) 
    
    # cur_fun: function evaluation, cur_grad: gradient evaluation
    # best_fun: best value of the objective function found so far
    cur_x = P0 # Starting point
    cur_fun = 0
    best_fun = 0
    best_x = P0
    cur_grad = 0

    while (previous_step_size > delta) & (iters < max_iters):
        # Evaluate fun and its gradient
        res = fun_first2last(cur_x,K,L,Multiplex,N_G,pi,pi_inv,w,W)
        cur_fun = -res[0]
        cur_grad = -res[1]
#        print(res)
        
        # Keep the best value and its variable
        if cur_fun > best_fun:
            best_fun = cur_fun
            best_x = cur_x
        
        # Update cur_x along the gradient and project it on the feasible set
        prev_x = cur_x
        cur_x = proj( cur_x + alpha * cur_grad , Pmax)
#        print(cur_x)
        previous_step_size = np.linalg.norm(cur_x - prev_x,ord=1)
        iters+=1

    # Evaluate the last point
    res = fun_first2last(cur_x,K,L,Multiplex,N_G,pi,pi_inv,w,W)
    cur_fun = -res[0]
    # Keep the best value and its variable
    if cur_fun > best_fun:
        best_fun = cur_fun
        best_x = cur_x

    # Round the last solution if needed
    if rounding_step is not None:
        best_x = proc_rounding(best_x,Pmax,rounding_step)

    best_fun = F_Pvect_first2last(best_x,K,L,Multiplex,N_G,pi,pi_inv,w,W)

    return best_x, best_fun, iters

# SpeedUp version of Grad_JSPA using SpeedUpSCUS
def SpeedUp_Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,max_iters=100,rounding_step=None):
    # Instantiate one SpeedUpSCUS object for each subcarrier
    # in order to have faster SCUS evaluation later
    suSCUS = [ SpeedUpSCUS(K,Multiplex,Pmax,N_G[lsub],lsub,pi,pi_inv,w,W) for lsub in range(L) ]
    
    # Initial value
    P0 = Pmax/L*np.ones(L)

    # Projected gradient descent with backtracking line search
    previous_step_size = math.inf
    iters = 0 #iteration counter
    # Step length: well chosen fixed step length alpha performs well
    # Backtracking line search can also be implemented but requires more computations
    alpha = 0.1/np.max(W)/np.max(w) 
    
    # cur_fun: function evaluation, cur_grad: gradient evaluation
    # best_fun: best value of the objective function found so far
    cur_x = P0 # Starting point
    cur_fun = 0
    best_fun = 0
    best_x = P0
    cur_grad = 0

    while (previous_step_size > delta) & (iters < max_iters):
        # Evaluate fun and its gradient
        res = fun_last2first(cur_x,suSCUS,K,L,Multiplex,N_G,pi,pi_inv,w,W)
        cur_fun = -res[0]
        cur_grad = -res[1]
        
        # Keep the best value and its variable
        if cur_fun > best_fun:
            best_fun = cur_fun
            best_x = cur_x
        
        # Update cur_x along the gradient
        prev_x = cur_x
        cur_x = proj( cur_x + alpha * cur_grad , Pmax)
#        print(cur_x)
        previous_step_size = np.linalg.norm(cur_x - prev_x,ord=1)
        iters+=1

    # Evaluate the last point
    res = fun_last2first(cur_x,suSCUS,K,L,Multiplex,N_G,pi,pi_inv,w,W)
    cur_fun = -res[0]
    # Keep the best value and its variable
    if cur_fun > best_fun:
        best_fun = cur_fun
        best_x = cur_x

    # rounding for the best found solution
    if rounding_step is not None:
        best_x = proc_rounding(best_x,Pmax,rounding_step)

    best_fun = np.zeros((L,K))
    for lsub in range(L):
        _, _, best_fun[lsub] = suSCUS[lsub].speedUp_SCUS(best_x[lsub])
        
    return best_x, np.transpose(best_fun), iters

# Basic heuristic which allocates equal power to each subcarrier
def eqPow_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W):
    P0 = Pmax/L*np.ones(L)
    val = F_Pvect_first2last(P0,K,L,Multiplex,N_G,pi,pi_inv,w,W)
    return P0, val

########################################### Fast and efficient JSPA Classes ###########################################

# epsilon_JSPA based on my TSP work: convert to MCKP and solve using a FPTAS
# espilon: approx ratio and error tolerance when inverting F^n (SCUS)
def epsilon_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,epsilon):
    # Instantiate one SpeedUpSCUS object for each subcarrier
    # in order to have faster SCUS evaluation later
    suSCUS = [ SpeedUpSCUS(K,Multiplex,min(PmaxN[lsub],Pmax),N_G[lsub],lsub,pi,pi_inv,w,W) for lsub in range(L) ]
    
    # For each subcarrier lsub, compute SCUS from 0 to min(Pmax,PmaxN[lsub]) 
    # such that two consecutive values differs at most epsilon*F^n(min(Pmax,PmaxN[lsub]))
    # Start with P=0 and P=min(Pmax,PmaxN[lsub]) then perform binary search to fill the list
    lists_power = [None] * L
    lists_wsr = [None] * L
    # precision: see TSP paper
    Fn_Pmax_over_N = np.zeros(L) 
    for lsub in range(L):
        Fn_Pmax_over_N[lsub] = suSCUS[lsub].speedUp_SCUS_onlyWSR(Pmax/L)
    precision = epsilon*np.sum(Fn_Pmax_over_N)/L
    for lsub in range(L):
        list_power = [0,min(Pmax,PmaxN[lsub])]
        list_wsr = [0,suSCUS[lsub].speedUp_SCUS_onlyWSR(min(Pmax,PmaxN[lsub]))]
        i = 0
        while True:
            if list_wsr[i+1] - list_wsr[i] <= precision:
                i += 1
                if i >= len(list_wsr)-1:
                    break
            else:
                middlePower = (list_power[i+1] + list_power[i])/2
                list_power.insert(i+1,middlePower)
                list_wsr.insert(i+1,suSCUS[lsub].speedUp_SCUS_onlyWSR(middlePower))
        lists_power[lsub] = list_power
        lists_wsr[lsub] = list_wsr
    
    # Normalize all wsr by precision 
    # then round down (floor) to integer 
    # then remove duplicate wsr values (and their corresponding power value)
    normalized_lists_wsr = [None] * L
    normalized_lists_power = [None] * L
    for lsub in range(L):
        normalized_lists_wsr[lsub], return_index = np.unique(np.floor(lists_wsr[lsub]/precision).astype(int),return_index=True)
        normalized_lists_power[lsub] = np.array(lists_power[lsub])[return_index]
#    print(normalized_lists_wsr[0])
#    print(normalized_lists_power[0])
#    print(np.sum(Fn_Pmax_over_N))
#    print(lists_wsr[0][-1])
    
    #------------------ FPTAS algorithm for MCKP using dynamic programming ------------------
    # countSCUS: count the number of SCUS call
    countSCUS = sum(len(x) for x in lists_wsr)
    # countNormWSR: count the number of normalized WSR
    countNormWSR = sum(len(x) for x in normalized_lists_wsr)
    # countDP: count the number of iterations (for loops) in the following DP
    countDP = 0
    # Dynamic programming to find the best power allocation given budget Pmax
    # DP[l,s]: lowest power budget to achieve wsr=s on the first l subcarriers
    # DP[l,s] = np.inf if wsr=s is not possible for budget less than Pmax
    DP = [None] * L
    # DP_index[lsub][i] stores the previous index (of lsub-1) chosen for this computation
    DP_index = [None] * L 
    # DPsize: sum of the first lsub normalized_lists_wsr last element, i.e. maximum size of the DP[lsub] array
    DPsize = 0 #sum(x[-1] for x in normalized_lists_wsr)+1
    for lsub in range(L):
        DPsize += normalized_lists_wsr[lsub][-1]
        DP[lsub] = np.ones(DPsize+1,dtype=int)*np.inf
        DP_index[lsub] = np.ones(DPsize+1,dtype=int)*(-1)
    # Initialization
    for index in range(len(normalized_lists_wsr[0])):
        cur_wsr = normalized_lists_wsr[0][index]
        cur_pow = normalized_lists_power[0][index]
        DP[0][cur_wsr] = cur_pow
#    print(DP[0])
        
    # DP recursion
    for lsub in range(1,L):
#        print(lsub)
        for i in range(len(normalized_lists_wsr[lsub])):
#            print('item',i)
            cur_wsr = normalized_lists_wsr[lsub][i]
            cur_pow = normalized_lists_power[lsub][i]
            for prev_wsr in range(len(DP[lsub-1])):
                prev_pow = DP[lsub-1][prev_wsr]
                if DP[lsub][prev_wsr+cur_wsr] > prev_pow+cur_pow:
                    DP[lsub][prev_wsr+cur_wsr] = prev_pow+cur_pow
                    DP_index[lsub][prev_wsr+cur_wsr] = prev_wsr
                # Increase countDP
                countDP += 1
    
    # Find the greatest index "opt_wsr" of DP such that DP[lsub][opt_wsr] <= Pmax
    # In the NOMA case, we simplify this to search only for lsub = L-1 (all subcarriers considered)
    opt_wsr = np.searchsorted(DP[L-1],Pmax)-1
    # Backtracking to get the power vector corresponding to opt_wsr
    x_cumul = np.zeros(L)
    pt_wsr = opt_wsr # backtracking pointer in the DP arrays as follows
    for lsub in range(L-1,-1,-1):
        x_cumul[lsub] = DP[lsub][pt_wsr]
        pt_wsr = DP_index[lsub][pt_wsr]
    # Get the non cumulative one
    x = np.ediff1d(x_cumul, to_begin=x_cumul[0])
#    print('TEST x',x)
#    print('sum x',np.sum(x))
#    print('opt_wsr',opt_wsr)
#    print('opt_wsr renormalized',opt_wsr*precision)
    # Project so that sum(x) == Pmax
    x = proj(x, Pmax)
    
    # Compute the final wsr vector
    final_wsr = np.zeros((L,K))
    for lsub in range(L):
        _, _, final_wsr[lsub] = suSCUS[lsub].speedUp_SCUS(x[lsub])
    
    return x, np.transpose(final_wsr), countSCUS, countNormWSR, countDP

# opt_JSPA based on my TSP work: convert to MCKP and solve using  
# the pseudo-polynomial dynamic programming by weights
# It performs computations on discretized subcarrier's budget at k*discrete_step 
def opt_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,discrete_step):
    # Instantiate one SpeedUpSCUS object for each subcarrier
    # in order to have faster SCUS evaluation later
    suSCUS = [ SpeedUpSCUS(K,Multiplex,min(PmaxN[lsub],Pmax),N_G[lsub],lsub,pi,pi_inv,w,W) for lsub in range(L) ]
    
    # For each subcarrier lsub, compute SCUS from 0 to min(Pmax,PmaxN[lsub]) with discrete_step
    lists_normalized_power = [None] * L
    lists_wsr = [None] * L
    for lsub in range(L):
        lists_normalized_power[lsub] = np.arange(0,round(min(Pmax,PmaxN[lsub])/discrete_step)+1,dtype=int)
        lists_wsr[lsub] = np.zeros(len(lists_normalized_power[lsub]))
        for i in range(len(lists_normalized_power[lsub])):
            lists_wsr[lsub][i] = suSCUS[lsub].speedUp_SCUS_onlyWSR(lists_normalized_power[lsub][i]*discrete_step)
    
    #------------------ FPTAS algorithm for MCKP using dynamic programming ------------------
    # Normalized Pmax
    normalized_Pmax = round(Pmax/discrete_step)
    # countSCUS: count the number of SCUS call
    countSCUS = sum(len(x) for x in lists_wsr)
    # countDP: count the number of basic operations in the following DP
    countDP = 0
    # Dynamic programming to find the best power allocation given budget Pmax
    # DP[l,b]: greatest wsr given budget b on the first l subcarriers
    DP = [None] * L
    # DP_index[lsub][i] stores the previous index (of lsub-1) chosen for this computation
    DP_index = [None] * L 
    # DPsize: sum of the first lsub normalized_lists_wsr last element, i.e. maximum size of the DP[lsub] array
    DPsize = 0 
    for lsub in range(L):
        DPsize += lists_normalized_power[lsub][-1]
        DP[lsub] = np.ones(min(DPsize,normalized_Pmax)+1)*(-1)
        DP_index[lsub] = np.ones(min(DPsize,normalized_Pmax)+1,dtype=int)*(-1)
    # Initialization
    DP[0] = lists_wsr[0]
#    print(DP[0])
        
    # DP recursion
    for lsub in range(1,L):
#        print(lsub)
        for i in range(len(lists_normalized_power[lsub])):
#            print('item',i)
            cur_wsr = lists_wsr[lsub][i]
            cur_pow = lists_normalized_power[lsub][i]
            for prev_pow in range(min(len(DP[lsub-1]),normalized_Pmax+1-cur_pow)):
                prev_wsr = DP[lsub-1][prev_pow]
                if DP[lsub][prev_pow+cur_pow] < prev_wsr+cur_wsr:
                    DP[lsub][prev_pow+cur_pow] = prev_wsr+cur_wsr
                    DP_index[lsub][prev_pow+cur_pow] = prev_pow
                # Increase count DP
                countDP += 1
    
    # Find the greatest index "opt_wsr" of DP such that DP[lsub][opt_wsr] <= Pmax
    # In the NOMA case, we simplify this to search only for lsub = L-1 (all subcarriers considered)
    opt_wsr = DP[L-1][normalized_Pmax]
    # Backtracking to get the power vector corresponding to opt_wsr
    x_cumul = np.zeros(L)
    pt_pow = normalized_Pmax # backtracking pointer in the DP arrays as follows
    for lsub in range(L-1,-1,-1):
        x_cumul[lsub] = pt_pow
        pt_pow = DP_index[lsub][pt_pow]
    # Get the non cumulative one
    x = np.ediff1d(x_cumul, to_begin=x_cumul[0])
    # Get the non normalized one
    x *= discrete_step
#    print('TEST x',x)
#    print('sum x',np.sum(x))
#    print('opt_wsr',opt_wsr)
#    print('opt_wsr renormalized',opt_wsr*discrete_step)
    # Project so that sum(x) == Pmax
    x = proc_rounding(x,Pmax,discrete_step)
    
    # Compute the final wsr vector
    final_wsr = np.zeros((L,K))
    for lsub in range(L):
        _, _, final_wsr[lsub] = suSCUS[lsub].speedUp_SCUS(x[lsub])
    
    return x, np.transpose(final_wsr), countSCUS, countDP

# Rounding procedure used in the previous functions
def proc_rounding(x,Pmax,rounding_step):
    # rounding
    steps = np.floor_divide(x,rounding_step)
    #    print('steps',steps)
    #    print(np.floor_divide(Pmax+rounding_step/3,rounding_step) - np.sum(steps))
    # Taking Pmax+rounding_step/3 to avoid precision issue converting float to int
    nb_remaining_steps = int(np.floor_divide(Pmax+rounding_step/3,rounding_step) - np.sum(steps))
    #    print(nb_remaining_steps)
    indices_to_increase = []
    if nb_remaining_steps > 0:
        indices_to_increase = np.argsort(x/rounding_step-steps)[-nb_remaining_steps:]  
    steps[indices_to_increase] += 1 
    return steps*rounding_step
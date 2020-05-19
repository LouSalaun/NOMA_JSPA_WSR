# NOMA_JSPA_WSR

A Python 3 implementation of the Joint Subcarrier and Power Allocation algorithms for WSR maximization in NOMA described in the following paper: [L. Salaün, M. Coupechoux, and C. S. Chen, "Joint Subcarrier and Power Allocation in NOMA: Optimal and Approximate Algorithms." IEEE Transactions on Signal Processing, vol. 68, pp.2215-2230, 2020](https://doi.org/10.1109/TSP.2020.2982786).

© 2016 - 2020 Nokia

Licensed under Creative Commons Attribution Non Commercial 4.0 International

SPDX-License-Identifier: CC-BY-NC-4.0

## Getting started

We first start by defining the following system parameters:
- `K` (integer) : number of users.
- `L` (integer) : number of subcarriers.
- `Multiplex` (integer) : number of users per subcarrier.
- `Radius` (float) : radius of the cell in meters.
- `rmin` (float) : min distance between a user and the BS.
- `G` (array of size (K,L)) : channel gains for each user on each subcarrier. This can be generated randomly following the above paper's propagation model using `G = generate_gains.generateGains(K,L,Radius,rmin)`.
- `B` (float) : total bandwidth.
- `W` (array of size L) : array of each subcarrier's bandwidth, usually: `W = np.ones(L)*B/L`.
- `N` (array of size KL) : array of noises for each user on each subcarrier. For example: `N = 10**((-174-30)/10)*B/L*np.ones(K*L)` corresponds to -174 dBm/Hz.
- `N_G` (array of size (L,K)) : noise over channel gains array, obtained by `N_G = ca_jspa.computeN_G(G,N)`.
- `pi` and `pi_inv` (array of size (L,K)) : decoding order and its inverse function obtained by `pi, pi_inv = ca_jspa.computePi(G,N)`.
- `w` (array of size K) : one weight for each user, which is used to define the weight sum-rate (WSR) objective function.
- `Pmax` (float) : the total cellular power budget available for DL transmission.
- `PmaxN` (array of size L) : power budget for each subcarrier (Note: this parameter is not used by algorithms Grad_JSPA, SpeedUp_Grad_JSPA and eqPow_JSPA).

The Joint Subcarrier and Power Allocation (JSPA) algorithms are implemented in "channel_allocation_JSPA.py". They all consider weighted sum-rate (WSR) as objective function. Below is a summary of the main algorithms:
- **Opt-JSPA:** 
  - Can be called using `opt_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,delta)`.
  - It finds the optimal WSR assuming the power allocation is discretized with precision delta. That is, the allocated powers can only take value of the form l\*delta, where l is in {0,1,...,floor(Pmax/delta)}.
  - Its computational complexity is in O((Pmax/delta)^2), disregarding K and L. 
- **epsilon-JSPA:** 
  - Can be called using `epsilon_JSPA(K,L,Multiplex,Pmax,PmaxN,N_G,pi,pi_inv,w,W,epsilon)`.
  - It computes an approximate solution that achieves at least (1-epsilon) times the optimal WSR. As a consequence, epsilon should be set between 0 and 1 excluded.
  - It is a FPTAS with computational complexity in O((1/epsilon)^2), disregarding K and L. 
- **Grad-JSPA:** 
  - Can be called using `SpeedUp_Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=None)`.
  - It is a heuristic based on the projected gradient method, where `delta` represents the error tolerance at termination, and `rounding_step` is the final power allocation rounding/discretization precision. By default `rounding_step = None` means that no rounding is performed on the final power allocation.
  - `Grad_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W,delta,rounding_step=None)` gives the same result but with larger running time since it uses the basic SCUS instead of SpeedUpSCUS as basic building block. Hence, `SpeedUp_Grad_JSPA` is preferred.
- **eqPow-JSPA:** 
  - Can be called using `eqPow_JSPA(K,L,Multiplex,Pmax,N_G,pi,pi_inv,w,W)`.
  - It is a simple heuristic that allocates equal power budget on each subcarrier, then it solves SCUS once for each subcarrier.

In summary, when looking for an optimal solution, **Opt-JSPA** should be used. In this case, `delta` is chosen according to the transmit power precision required by the system (for example 10 mW for a total power budget of 1W). **epsilon-JSPA** provides an approximate solution. It has lower complexity than **Opt-JSPA** as long as `epsilon` is not too small. Otherwise, **Opt-JSPA** should be used instead. **Grad-JSPA** and **eqPow-JSPA** can be used to obtain low complexity solutions with good performance in average (without any performance guarantee). These JSPA algorithms are shown in file "example_JSPA.py".

Some examples of the single-carrier algorithms SCPC and SCUS can be found in "example_SCPC.py" and "example_SCUS.py". These two algorithms are mainly used as building blocks inside the JSPA algorithms.

#!/usr/bin/env python
# coding: utf-8

# In[71]:


# some installations
import sympy as sp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LogNorm, Normalize
from sympy.printing.mathematica import mathematica_code
from sympy.parsing.mathematica import parse_mathematica
import scipy
from lindbladian import *

import matplotlib as mpl
mpl.rcParams.update(mpl.rcParamsDefault) # needed because latex plots can't be rendered on shemesh


# In[66]:


# simulation parameters
time_start = 0
time_stop = 10
time_step = 1
times = np.arange(time_start, time_stop, time_step)

# constants
J_val = 0.1
epsilon_val = 1e-4

# number of sites
N = 6 

# initial position
psi_0 = np.zeros(2**N)
# site 1 initially in equal superposition
psi_0[2**(N-1)]=1/np.sqrt(2)
psi_0[0]=1/np.sqrt(2)
# site 1 initially excited
'''
psi_0[2**(N-1)]=1
'''

J = sp.Symbol("J", real=True, positive=True)
epsilon = sp.Symbol("epsilon", real=True, positive=True)
gamma = sp.Symbol("Gamma", real=True, positive=True)

# building system hamiltonian
H_flipflop = sp.zeros(2**N, 2**N)
for j in range(N):
    for i in range(j):
        coeff = J if (i==0 and j==1) else epsilon
        H_flipflop += coeff*(sigma_plus(i,N)*sigma_minus(j,N)+sigma_plus(j,N)*sigma_minus(i,N))


# In[67]:


Hz = sp.zeros(2**N, 2**N)
for j in range(N):
    for i in range(j):
        coeff = J if (i==0 and j==1) else epsilon
        Hz += coeff*(sigma(sigma_z, i, N)*sigma(sigma_z, j, N))


# In[68]:


H_full = H_flipflop + Hz


# In[72]:


# buildling matrices
sigma_z_1=np.array(sigma(sigma_z, 0, N)).astype(complex)
sigma_x_1=np.array(sigma(sigma_x, 0, N)).astype(complex)

# time evolution
def psi(t, H, psi_0):
    return scipy.linalg.expm(-1.0j*np.array(H).astype(complex)*t)@psi_0


# In[73]:


# subbing values into H
H = H_full.subs({J:J_val, epsilon:epsilon_val})

# setting up fig
fig, axs = plt.subplots(1,2, figsize=(12,5), layout="constrained")


# computing pops and cohs
pop1 = [-1*psi(time, H, psi_0).conj().transpose()@sigma_z_1@psi(time, H, psi_0) for time in times]
coh1 = [psi(time, H, psi_0).conj().transpose()@sigma_x_1@psi(time, H, psi_0) for time in times]


# plotting
axs[0].plot(times, pop1, label="100")
axs[0].set_xlabel("Time (s)")
#axs[0].legend(title="Site:")
axs[0].set_title("Population of Site 1")
axs[1].plot(times, np.real(coh1), label="Real")
axs[1].plot(times, np.imag(coh1), label="Imag")
axs[1].set_xlabel("Time (s)")
axs[1].legend()
#axs[1].ticklabel_format(axis='x', style='sci', scilimits=(0, 0)) 
axs[1].set_title("Coherence of Site 1")

fig.suptitle("Initally Site 1 in equal superposition" + "\n" + f"N={N}, J={J_val}, epsilon={epsilon_val}, t in [{time_start}, {time_stop}], delta t = {time_step}")

fig.savefig(f"N={N};J={J_val};epsilon={epsilon_val};[{time_start},{time_stop}];deltat={time_step}.pdf")

# saving as .dat files
np.savetxt( f"N={N};J={J_val};epsilon={epsilon_val};[{time_start},{time_stop}];deltat={time_step}.dat", np.array([times, pop1, coh1]).transpose() )


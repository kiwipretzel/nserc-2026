#!/usr/bin/env python
# coding: utf-8

# In[1]:


# some installations
import sympy as sp
from sympy.physics.quantum.dagger import Dagger
from sympy.physics.quantum import TensorProduct
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import LogNorm, Normalize
from sympy.printing.mathematica import mathematica_code
from sympy.parsing.mathematica import parse_mathematica
import scipy
import scienceplots


# In[39]:


# allows for Latex printing in cell output
sp.init_printing()

# increases fig resolution (if saving as png, jpeg, etc...)
plt.rcParams['figure.dpi'] = 400

# defining some useful, symbols
t = sp.Symbol('t')
h_bar = sp.Symbol("hbar")

# specifying to use the 'science' style for all plots
plt.style.use('science')


# In[38]:


# useful functions...

def commutator(A,B):
    return A@B - B@A

def anti_commutator(A,B):
    return A@B + B@A

def L_i_j(i,j, N): # Lindbladian jump operators... Lij = |j><i|
    return sp.Matrix(N, N, lambda m, n: 1 if (n==i and m==j) else 0) 


# In[37]:


def lindbladian(N=3, output=False, degenerate=False, H=sp.Matrix([0]), L_operators=[], dissipative=True, damping_rates=[]):
    ''' 
    this function builds the Lindbladian/Louisvillian superoperator in matrix form in the local/site basis
    (row-major convention is used for vectorization of the density matrix)

    inputs:
    N - number of sites in system
    output - whether you want intermediate matrices (e.i. V, H, D, etc...) to be displayed
    degenerate - if True, will automatically set site energies to 0
    H - you can specify your own Hamiltonian; if not specified, function builds a completely generic H
    L_operators - you can specify your own Lindbladian jump operators; if not specified, complex generic operators are used
    dissipative - if False, sets rate constants of dissipator to 0 (e.i. the returned matrix then represents unitary time evolution)
    damping_rates - a list you can provide that specifies the values of the damping rates, the ith element of damping_rates is the 
                damping rate for the ith element of L_operators; if not specified, completely generic damping_rates's are used
    '''

    # unitary time evolution 
    # if H is not specified, we build a completely generic H
    if H == sp.Matrix([0]):
        H0 = sp.Matrix(N, N, lambda i, j: sp.Symbol(f"epsilon{i}{j}") if i==j else 0) # unperturbed hamiltonian

        # setting levels degenerate
        if degenerate: H0 = H0.subs({f"epsilon{i}{i}":0 for i in range(N)})

        V = sp.Matrix(N, N, lambda i, j: sp.Symbol(f"V{i}{j}") if i!=j else 0) # perturbation

        # since V is hermitian
        V = V.subs({sp.Symbol(f"V{j}{i}"):sp.Symbol(f"V{i}{j}").conjugate() for i in range(N) for j in range(i)})

        H = H0 + V

    # if H is specified, we set N to the dim of H
    else: N = sp.shape(H)[0]

    # building our density matrix
    rho = sp.Matrix(N, N, lambda i, j: sp.Function(f"rho{i}{j}")(t)) if N<=11 else sp.Matrix(N, N, lambda i, j: sp.Function(f"rho_{i},{j}")(t))

    # if user wants to see intermediate output, then display H
    if output: H

    # computing dissipator only if dissipative is True, otherwise the dissipator is 0
    D = dissipator(N, L_operators, damping_rates) if dissipative else sp.zeros(N)

    # if user wants to see intermediate output, then display D
    if output: display(D)

    # computing lindbladian as the sum of the unitary and dissipative matrices
    P = (-sp.I*commutator(H,rho) + D)

    # solving for the Lindbladian/Louisvillian superoperator matrix
    eqs = [P[i, j] for i in range(N) for j in range(N)] 
    rho_vec = list(rho.reshape(1, N**2)) # building a vectorized rho (using row-major convention)
    L, _ = sp.linear_eq_to_matrix(eqs, rho_vec)

    # reshaping rho_vec into a column vector
    rho_vec = sp.Matrix(rho_vec).reshape(N**2,1)

    # if user wants to see intermediate output, then display L
    if output: L

    # returning the Lindbladian/Louisvillian superoperator matrix and vectorized density matrix
    return L, rho_vec


def dissipator(N, L_operators, damping_rates):
    '''
    this function building the dissipator matrix in the site/local basis

    inputs:
    N - number of sites in system
    L_operators - a list of jump operators to use to compute the dissipator.
        If the list is empty, completely generic jump operators are used.
    damping_rates - a list you can provide that specifies the values of the damping rates, the ith element of damping_rates is the 
            damping rate for the ith element of L_operators; if not specified, completely generic damping_rates's are used
    '''

    # building density matrix
    rho = sp.Matrix(N, N, lambda i, j: sp.Function(f"rho{i}{j}")(t))

    # initializing dissipator matrix
    D = sp.zeros(N,N)

    # checking to see if any jump operators were provided
    if L_operators:

        # if damping rates were provided, we need to make sure enough were provided
        if damping_rates and np.shape(L_operators)[0] != np.shape(damping_rates)[0]:
            print("You did not provide the same number of L_operators and damping_rates's\nReturning a dissipator of 0")
            return D

        # enough jump operators were provided, so we just use the provided operators
        for i, operator in enumerate(L_operators):
            rate_const_gamma = damping_rates[i] if damping_rates else sp.Symbol(f"Gamma_{i}", real=True, nonnegative=True)
            D += rate_const_gamma*(operator*rho*Dagger(operator) - anti_commutator(Dagger(operator)*operator, rho)*sp.Rational(1,2))

    else:
        # no jump operators were provided, so we use completely generic operators Lij=|j><i| for all i and j

        # population transitions: i->j for i not equal j
        for i in range(N):
            for j in range(N):
                if i != j: # avoids double counting
                    rate_const_gamma = sp.Symbol(f"Gamma_{i}→{j}", real=True, nonnegative=True)
                    D += rate_const_gamma*(L_i_j(i,j,N)*rho*Dagger(L_i_j(i,j,N)) - anti_commutator(Dagger(L_i_j(i,j,N))*L_i_j(i,j,N), rho)*sp.Rational(1,2))

        # dephasing: i->i
        for i in range(N):
            rate_const_gamma = sp.Symbol(f"Gamma_{i}→{i}", real=True, nonnegative=True)
            D += rate_const_gamma*(L_i_j(i,i,N)*rho*Dagger(L_i_j(i,i,N)) - anti_commutator(Dagger(L_i_j(i,i,N))*L_i_j(i,i,N), rho)*sp.Rational(1,2))


    # returns the dissipator matrix      
    return D


# In[36]:


def solve_lindblad_ana(L, rho, initial_conditions = [], t0=0):
    '''
    given a set of lindbladian equations and initial conditions (at t=t0), this function analytically solves for each component of the density matrix
    more generically, it solves for a vector v that satisfies the matrix equation: d/dt v = M*v

    inputs:
    L - lindbladian/louisvillian superoperator matrix in site/local basis
    rho - density matrix (in vectorized form)
    initial_coniditions - a list where the ith component is the initial condition for the ith component in rho.
        if this list is empty, the equations are solved for with generic initial conditions

    returns:
    list containing analytic solutions to the matrix equation
    '''
    # checking to see if initial conditions were provided
    if not initial_conditions:
        # since no initial conditions were provided, we just use generic initial conditions
        initial_conditions = [rho[i].subs(t, t0) for i in range(len(rho))]

    # solving the system of equations analytically
    return sp.dsolve([sp.Eq(rho.diff()[i], (L@rho)[i]) for i in range(len(rho))], 
                     ics={rho[i].subs(t, t0): initial_conditions[i] for i in range(len(rho))},
                    hint="best",
                     x0=0,
                     simplify=False
                    )



def compute_eq_ana(expr, independent_variable, start, stop, delta):
    '''
    given a sympy expression with only one independent and symbolic variable, this function evaluates the
    expression at different values of the independent variable

    inputs:
    expr - symbolic sympy expression with only one symbol
    independent_variable - sympy symbol that you want to vary
    start - initial value of independent_variable
    stop - final value of independent_variable
    delta - increments of the independent_variable

    returns:
    Numpy array containing the different values of indepent_variable, Numpy array with the expr evaluations
    '''

    times = np.arange(start, stop, delta)

    f = sp.lambdify(independent_variable, expr) # lambdifying the expression to allow to fast Numpy or SciPy computions

    return times, np.full_like(times, f(times.astype(complex)), dtype=complex)



# In[1]:


def solve_lindblad_num(L, start, stop, dt, initial_conditions):
    '''
    given a set of lindbladian equations and initial conditions (at t=0), this function solves for each component of the density matrix
    more generically, it solves for a vector v that satisfies the matrix equation: d/dt v = M*v

    inputs:
    L - lindbladian/louisvillian superoperator matrix in site/local basis
    start - time you want to start the numeric solution
    stop - time you want to stop the numeric solution
    dt - increment of time step
    initial_coniditions - a list where the ith component is the initial condition for the ith component in rho.

    returns:
    Numpy array contains times, Numpy array where the ith element contains an array with the ith solution of rho
    '''

    times = np.arange(start, stop, dt)

    vals = [scipy.linalg.expm(np.array(L).astype(complex)*time)@np.array(initial_conditions) for time in times]

    return times, np.array(vals).T


# In[34]:


def plot_spectrum(expr, independent_variable, start, stop, delta, color="viridis", label="", size=2, norm="linear", ax="", fig="", title=""):
    '''
    given a symbol sympy expression, this function evaluates the expr at different values of the independent
    variable and then plots the value of the expression in the complex plane

    inputs:
    expr - symbolic sympy expression as a function of only one symbol
    indepent_variable - string of character acting as independent variable
    start - initial value of independent_variable
    stop - final value of independent_variable
    delta - increment of independent_variable
    color - color scheme to use for color bar of independent variable (defaults "viridis")
    label - label to be attached to color bar of independent variable (defaults "")
    size - size of markers
    norm - scaling to use on the color bar (usually use "liner" or "log")
    ax, fig - you choose provide your own matplotlib axes and figures; if not specified, function creates its own
    title - title of plot
    '''

    # creating own axes and figures if none were provided
    if not ax or not fig:
        fig, ax = plt.subplots()

    steps = np.arange(start, stop, delta) if norm=="linear" else np.logspace(np.log10(start), np.log10(stop), 200)
    scale = LogNorm(vmin=start, vmax=stop) if norm=="log" else Normalize(vmin=start, vmax=stop)

    f = sp.lambdify(sp.Symbol(independent_variable), expr)


    val = f(steps.astype(complex))
    ax.scatter(np.real(val), np.imag(val), s=size, c=steps, cmap=color, label=label, norm=scale)


    fig.colorbar(cm.ScalarMappable(norm=scale, cmap=color), ax=ax).set_label(label)
    ax.grid()
    ax.set_title(title)
    ax.set_axisbelow(True)
    ax.set_xlabel("Real Axis")
    ax.set_ylabel("Imag Axis")


# In[2]:


# defining pauli matrices
sigma_x = sp.Matrix([[0, 1], [1, 0]])
sigma_y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
sigma_z = sp.Matrix([[1, 0], [0, -1]])
sigma_0 = sp.eye(2)

def sigma(sigma, i, n):
    '''
    given a pauli matrix (sigma), this function returns the corresponding matrx acting on site i out of n sites

    inputs:
    sigma - pauli matrix that you want in higher dimensional hilbert space
    i - which site you want the matrix to act on (0 indexed)
    n - total number of sites in the space

    returns:
    pauli matrix acting on site i out of n sites

    e.g. if sigma=sigma_z, i=1, n=4, then:
    returns I ⊗ sigma_z ⊗ I ⊗ I
    '''

    chain = [sigma_0 for i in range(i)]

    chain.append(sigma)

    chain.extend([sigma_0 for i in range(n-i-1)])

    return TensorProduct(*chain)


def sigma_plus(i, n):
    '''
    returns the sigma_plus (e.i. raising) operator that acts on site i out of n total sites (0 index)
    sigma_plus = (sigma_x + i sigma_y)/2
    '''
    return (sigma(sigma_x, i, n) + sp.I*sigma(sigma_y, i, n))*sp.Rational(1/2)


def sigma_minus(i, n):
    '''
    returns the sigma_minus (e.i. lowering) operator that acts on site i out of n total sites (0 index)
    sigma_minus = (sigma_x - i sigma_y)/2
    '''
    return (sigma(sigma_x, i, n) - sp.I*sigma(sigma_y, i, n))*sp.Rational(1/2)


# In[ ]:





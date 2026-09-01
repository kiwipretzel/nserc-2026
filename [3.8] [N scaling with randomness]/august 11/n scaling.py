#!/usr/bin/env python
# coding: utf-8

# In[2]:


# retrieving functions from lindbladian.ipnyb
get_ipython().run_line_magic('run', "'/home/jovyan/NSERC 2026/[1] [lindbladian helper code] [may 19]/lindbladian.ipynb'")


# In[2]:


# using 'Science' style plots
plt.style.use(['science', 'high-vis'])

J = sp.Symbol("J", real=True, positive=True)
epsilon  = sp.Symbol("epsilon", real=True, positive=True)
gamma = sp.Symbol("Gamma", real=True, positive=True)


# In[8]:


def build_L(N, num_dimers=1, output=False):
    # building matrices
    H0 = sp.Matrix(N, N, lambda i,j : J if ({i,j} in [{2*k, 2*k+1} for k in range(num_dimers)]) else 0)
    V = sp.Matrix(N, N, lambda i,j : epsilon*sp.Symbol(f"epsilon_{min(i,j)}{max(i,j)}", real=True, positive=True) 
                  if ({i,j} not in [{2*k, 2*k+1} for k in range(num_dimers)] and i!=j) else 0)

    if output: display(H0+V)

    # building lindblad jump operators
    L_operators = [sp.Matrix(N,N, lambda i,j: 1 if i==k and j==k else 0) for k in range(N)]

    # building lindbladians
    L, rho = lindbladian(N, output=False, degenerate=False, H=H0+V, L_operators=L_operators, damping_rates=[gamma for i in range(N)])

    return L, rho


# In[9]:


def randomize(N, num_dimers, matrix, seed=-1, low=1, high=10):
    rng = np.random.default_rng(seed=seed) if seed != -1 else np.random.default_rng()

    epsilonij_val = { sp.Symbol(f"epsilon_{min(i,j)}{max(i,j)}", real=True, positive=True):rng.uniform(low,high)
  for i in range(N) for j in range(i) 
  if ({i,j} not in [{2*k, 2*k+1} for k in range(num_dimers)] and i!=j) }

    return matrix.subs(epsilonij_val), epsilonij_val


# In[10]:


def find_liousvillian_gap(matrix, atol=1e-17):
    # if converting to numpy just in case:
    matrix = np.array(matrix).astype(complex)

    # finding evals
    evals = scipy.linalg.eigvals(matrix)

    # uses a mask to eliminate the 0 eigenvalue (may have to adjust tolerance),
    # then finds the min of the magnitudes of the real parts of the non-zero evals
    return np.min(np.abs(np.real(evals)[~np.isclose(np.real(evals),np.zeros(np.shape(matrix)[0]), atol=atol)]))


# In[11]:


def gaps_of_liousvillian(iterations=3, N=3, num_dimers=1, J_val=0.1, gamma_val=1e-6, epsilon_val=1e-4, seed=-1, low=1, high=10):
    print(f"Starting N={N}: ", end="")

    # generating our seeds that we will use for each iteration
    rng = np.random.default_rng(seed=seed) if seed != -1 else np.random.default_rng()
    seeds = rng.integers(1e3, size=iterations)

    gaps = []

    # building the Linbladian
    L, rho = build_L(N, num_dimers=num_dimers)
    # plugging in our known vals
    L = L.subs({J:J_val, gamma:gamma_val, epsilon: epsilon_val})


    # iterating through each seed and computing the liousvillian gap
    for i in range(iterations):
        print(i, end=", ")

        # randomizing L
        L_random, epsilonij_val = randomize(N, num_dimers, L, seed=seeds[i], low=low, high=high)
        gaps.append(find_liousvillian_gap(L_random))

    print()


    return gaps








# In[63]:


def linear_fit(fig, ax, x, y):
    # linear fit of the form: y=mx+b
    m, b, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    ax.plot(x,m*x+b, color="gray", linestyle="--", alpha=0.5, label="linear fit")
    ax.annotate(fr"$y={np.real(m):.4f}x{"+" if np.real(b)>0 else "-"}{np.abs(b):.4f}$"+ "\n" + fr"$R^2 = {np.real(r_value)**2:.4f}$",
                         (0.1,0.8),
                         xycoords="axes fraction",
                         fontsize="14",
                         ha="left"
                        )
    return m, b, r_value**2, std_err


# In[ ]:


# NETWORK PARAMETERS
J_val = 0.1
epsilon_val = 1e-4
gamma_val = 1e-6
seed = 15
num_dimers = 1
high = 1
low = 0

# SAVING PARAMETERS
save_data = True
save_fig = True
path_data = "data"
path_fig = "fig"


# SIMULATION PARAMETERS
N_start = 3
N_stop = 10
iterations_per_N = 1000

fig, ax = plt.subplots()

# starting simulation
gaps = np.asarray([ gaps_of_liousvillian(iterations=iterations_per_N,
                                         N=i, 
                                         num_dimers=num_dimers, 
                                         J_val=J_val, 
                                         gamma_val=gamma_val, 
                                         epsilon_val=epsilon_val, 
                                         seed=seed,
                                         high=high,
                                         low=low,
                                ) 
         for i in range(N_start, N_stop+1) ]
)

if save_data:
    np.savetxt(f"{path_data}/J={J_val}; epsilon={epsilon_val}; gamma={gamma_val}; seed={seed}; iterations={iterations_per_N}; num_dimers={num_dimers}; low={low}; high={high}.csv", np.insert(gaps, 0, x, axis=1), delimiter=",")

x = np.arange(N_start, N_stop+1)



# PLOTTING
gaps = gaps / (2 * gamma_val * epsilon_val**2 / (gamma_val**2 + J_val**2) )
y = np.mean(gaps, axis=1) 
yerr = np.std(gaps, axis=1)
ax.scatter(x, y)
ax.errorbar(x, y, label="data", yerr=yerr, fmt="o")
linear_regression(fig, ax, x, y)
ax.set_xlabel(r"$N$")
ax.set_ylabel(r"$ \Delta \div \frac{ 2 \Gamma \epsilon^2 }{\Gamma^2 + J^2}$")
ax.set_title(rf"Smallest eigenvalue ($\Delta$) averaged from {iterations_per_N} networks of size $N$" + "\n" + rf" with randomized $\epsilon_i \sim U({{{low}}}, {{{high}}})$" + "\n" + rf"$J=10^{{{int(np.log10(J_val))}}}; \epsilon=10^{{{int(np.log10(epsilon_val))}}}; \Gamma=10^{{{int(np.log10(gamma_val))}}}$; seed={{{seed}}}")
ax.legend()



if save_fig:
    fig.savefig(f"{path_fig}/J={J_val}; epsilon={epsilon_val}; gamma={gamma_val}; seed={seed}; iterations={iterations_per_N}; num_dimers={num_dimers}; low={low}; high={high}.svg")




# In[ ]:


fig


# In[106]:


fig.savefig(f"{path_fig}/J={J_val}; epsilon={epsilon_val}; gamma={gamma_val}; seed={seed}; iterations={iterations_per_N}; num_dimers={num_dimers}; low={low}; high={high}.svg")



# In[ ]:





# In[7]:


get_ipython().system('jupyter nbconvert n scaling.ipynb --TagRemovePreprocessor.enabled=True --TagRemovePreprocessor.remove_cell_tags="[\'ipnyb only\']" --to script')


# In[ ]:





# In[ ]:





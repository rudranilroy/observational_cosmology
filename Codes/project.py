#!/usr/bin/env python
# coding: utf-8

# # Import the packages

# In[ ]:


import pandas as pd
import numpy as np
from scipy.integrate import quad
import scipy.linalg as la
import matplotlib.pyplot as plt

import emcee
import corner
import os
import time
from multiprocess import Pool, cpu_count


# # Load All Datasets (SNe, CC, BAO)

# In[ ]:


# ---------------------------------------------------------
# Load All Datasets and Covariance Matrices
# ---------------------------------------------------------

# 1. Pantheon+SH0ES SNe
df_sn = pd.read_csv('data/Pantheon+SH0ES.dat', sep=r'\s+')
N_sn = len(df_sn)
with open('data/Pantheon+SH0ES_STAT+SYS.cov', 'r') as f:
    lines = f.readlines()
if len(lines[0].split()) == 1: lines = lines[1:]
C_sn = np.loadtxt(lines).reshape((N_sn, N_sn))
c_chol_sn, lower_sn = la.cho_factor(C_sn)
C_inv_sn = la.cho_solve((c_chol_sn, lower_sn), np.eye(N_sn))
log_det_C_sn = 2.0 * np.sum(np.log(np.diag(c_chol_sn)))

# 2. Cosmic Chronometers (CC)
df_cc = pd.read_csv('data/CC.txt', sep=r'\s+')

# 3. DESI BAO Data
df_bao = pd.read_csv('data/desi_gaussian_bao_ALL_GCcomb_mean.txt', sep=r'\s+', comment='#', names=['z', 'measurement', 'observable_type'])
N_bao = len(df_bao)
C_bao = np.loadtxt('data/desi_gaussian_bao_ALL_GCcomb_cov.txt').reshape((N_bao, N_bao))
c_chol_bao, lower_bao = la.cho_factor(C_bao)
C_inv_bao = la.cho_solve((c_chol_bao, lower_bao), np.eye(N_bao))

print("All datasets loaded successfully.")


# # CPL Dynamical Dark Energy Physics Engine

# In[ ]:


# ---------------------------------------------------------
# Physics Engine: CPL Parametrization
# ---------------------------------------------------------
c = 299792.458 # km/s

def E_inverse(z, Omega_m, w0, wa):
    term_matter = Omega_m * (1 + z)**3
    exponent = 3 * (1 + w0 + wa)
    exponential_term = np.exp(-3 * wa * (z / (1 + z)))
    term_de = (1.0 - Omega_m) * ((1 + z)**exponent) * exponential_term
    return 1.0 / np.sqrt(term_matter + term_de)

def comoving_distance_mpc(z, H_0, Omega_m, w0, wa):
    integral, _ = quad(E_inverse, 0, z, args=(Omega_m, w0, wa))
    return (c / H_0) * integral

def luminosity_distance_mpc(z_hel, z_cmb, H_0, Omega_m, w0, wa):
    return (1 + z_hel) * comoving_distance_mpc(z_cmb, H_0, Omega_m, w0, wa)

def model_cosmology_sn(z_hel, z_cmb, M_B, H_0, Omega_m, w0, wa):
    d_L = luminosity_distance_mpc(z_hel, z_cmb, H_0, Omega_m, w0, wa)
    return M_B + 5.0 * np.log10(d_L) + 25.0

def H_CPL(z, Omega_m, H_0, w0, wa):
    return H_0 / E_inverse(z, Omega_m, w0, wa)

def D_M(z, H_0, Omega_m, w0, wa):
    return comoving_distance_mpc(z, H_0, Omega_m, w0, wa)

def D_H(z, H_0, Omega_m, w0, wa):
    return c / H_CPL(z, Omega_m, H_0, w0, wa)

def D_V(z, H_0, Omega_m, w0, wa):
    dm = D_M(z, H_0, Omega_m, w0, wa)
    dh = D_H(z, H_0, Omega_m, w0, wa)
    return np.cbrt(z * dm**2 * dh)

def bao_theory(z, obs_type, H_0, Omega_m, r_d, w0, wa):
    if obs_type == 'DV_over_rs': return D_V(z, H_0, Omega_m, w0, wa) / r_d
    elif obs_type == 'DM_over_rs': return D_M(z, H_0, Omega_m, w0, wa) / r_d
    elif obs_type == 'DH_over_rs': return D_H(z, H_0, Omega_m, w0, wa) / r_d
    else: raise ValueError(f"Unknown BAO type: {obs_type}")


# # Dynamic Likelihoods

# In[ ]:


# ---------------------------------------------------------
# Dynamic Log-Likelihood Definitions
# ---------------------------------------------------------
def log_prior(params):
    Omega_m, H_0, M_B, r_d, w0, wa = params

    # 1. Broad physical bounds
    bounds = (
        (0.01 < Omega_m < 0.99) and (50.0 < H_0 < 100.0) and 
        (-21.0 < M_B < -18.0) and (120.0 < r_d < 170.0) and
        (-3.0 < w0 < 1.0) and (-3.0 < wa < 3.0)
    )
    if not bounds: return -np.inf

    # 2. External SH0ES Prior for M_B
    lp_MB = -0.5 * ((M_B - (-19.253)) / 0.027)**2
    return lp_MB

def log_likelihood_sn(params, data_sn, C_inv, log_det_C):
    Omega_m, H_0, M_B, _, w0, wa = params
    delta = np.zeros(len(data_sn))
    for i, row in data_sn.iterrows():
        m_b_th = model_cosmology_sn(row['zHEL'], row['zCMB'], M_B, H_0, Omega_m, w0, wa)
        delta[i] = row['m_b_corr'] - m_b_th
    chi2 = np.dot(delta.T, np.dot(C_inv, delta))
    return -0.5 * (chi2 + log_det_C + len(data_sn) * np.log(2 * np.pi))

def log_likelihood_cc(params, data_cc):
    Omega_m, H_0, _, _, w0, wa = params
    chi2 = sum(((row['Hz'] - H_CPL(row['z'], Omega_m, H_0, w0, wa)) / row['err_Hz'])**2 for _, row in data_cc.iterrows())
    return -0.5 * chi2

def log_likelihood_bao(params, data_bao, C_inv_bao):
    Omega_m, H_0, _, r_d, w0, wa = params
    delta = np.zeros(len(data_bao))
    for i, row in data_bao.iterrows():
        th_val = bao_theory(row['z'], row['observable_type'], H_0, Omega_m, r_d, w0, wa)
        delta[i] = row['measurement'] - th_val
    chi2 = np.dot(delta.T, np.dot(C_inv_bao, delta))
    return -0.5 * chi2

def log_posterior_master(params, use_sn, use_cc, use_bao):
    lp = log_prior(params)
    if not np.isfinite(lp): return -np.inf

    ll = 0.0
    if use_sn: ll += log_likelihood_sn(params, df_sn, C_inv_sn, log_det_C_sn)
    if use_cc: ll += log_likelihood_cc(params, df_cc)
    if use_bao: ll += log_likelihood_bao(params, df_bao, C_inv_bao)

    return lp + ll


# # The Master MCMC & Plotting Function (Updated with Save Paths)

# In[ ]:


# ---------------------------------------------------------
# Master Automation Function for all 7 Runs
# ---------------------------------------------------------
os.environ["OMP_NUM_THREADS"] = "1"

# Create the target directories if they don't exist
os.makedirs("mcmc_chains/project", exist_ok=True)
os.makedirs("plots/project", exist_ok=True)

def compute_gelman_rubin(chain):
    nsteps, nwalkers, ndim = chain.shape
    R_hat = np.zeros(ndim)
    for d in range(ndim):
        samples = chain[:, :, d]
        W = np.mean(np.var(samples, axis=0, ddof=1))
        chain_means = np.mean(samples, axis=0)
        grand_mean = np.mean(chain_means)
        B = (nsteps / (nwalkers - 1.0)) * np.sum((chain_means - grand_mean)**2)
        V_hat = ((nsteps - 1.0) / nsteps) * W + (B / nsteps)
        R_hat[d] = np.sqrt(V_hat / W) if W > 0 else np.inf
    return R_hat

def run_analysis(run_name, title, use_sn, use_cc, use_bao):
    print(f"\n{'='*60}\nSTARTING RUN: {run_name}\n{'='*60}")

    ndim = 6
    nwalkers = 48
    max_steps = 15000
    check_interval = 200
    ncpu = int(max(1, cpu_count() / 2 - 1))

    # Init guess
    initial_pos = np.array([0.31, 70.0, -19.24, 147.0, -1.0, 0.0]) + 1e-3 * np.random.randn(nwalkers, ndim)

    # --- UPDATED CHAIN SAVE PATH ---
    safe_name = run_name.replace(' ', '_')
    filename = f"mcmc_chains/{safe_name}.h5"

    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    # 1. RUN MCMC
    start_time = time.time()
    with Pool(processes=ncpu) as pool:
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_posterior_master, 
            args=(use_sn, use_cc, use_bao), pool=pool, backend=backend
        )
        for sample in sampler.sample(initial_pos, iterations=max_steps, progress=True):
            if sampler.iteration % check_interval == 0 and sampler.iteration > 0:
                chain = sampler.get_chain()
                burn_in = int(chain.shape[0] * 0.5)
                R_hat = compute_gelman_rubin(chain[burn_in:, :, :])
                if np.all(R_hat < 1.02):
                    print(f"\n✅ Convergence achieved at step {sampler.iteration}!")
                    break

    print(f"Time taken: {(time.time() - start_time) / 60:.2f} min.")

    # 2. EXTRACT DATA
    total_steps = sampler.iteration
    discard = int(total_steps * 0.5)
    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)
    chain = sampler.get_chain() # Full chain for trace
    labels = [r"$\Omega_m$", r"$H_0$", r"$M_B$", r"$r_d$", r"$w_0$", r"$w_a$"]

    # 3. TRACE PLOT
    fig, axes = plt.subplots(ndim, figsize=(10, 12), sharex=True)
    for i in range(ndim):
        axes[i].plot(chain[:, :, i], "k", alpha=0.3)
        axes[i].set_ylabel(labels[i], fontsize=14)
        axes[i].axvline(x=discard, color='red', linestyle='--')
    axes[-1].set_xlabel("Step Number", fontsize=12)
    plt.suptitle(f"Trace Plot: {title}", y=1.02, fontsize=16)
    plt.tight_layout()

    # --- UPDATED TRACE PLOT SAVE PATH ---
    plt.savefig(f"plots/project/{safe_name}_trace.png", dpi=300, bbox_inches='tight')
    plt.show()

    # 4. CORNER PLOT
    fig = corner.corner(
        flat_samples, labels=labels,
        quantiles=[0.16, 0.5, 0.84], show_titles=True,
        title_kwargs={"fontsize": 12}, color='teal'
    )
    plt.suptitle(f"Posterior: {title}", y=1.02, fontsize=16)

    # --- UPDATED CORNER PLOT SAVE PATH ---
    plt.savefig(f"plots/project/{safe_name}_corner.png", dpi=300, bbox_inches='tight')
    #plt.show()


# # Run Phase 1 (Individual Analyses)

# In[ ]:


# 1. SNe Only
run_analysis(
    "Run_1_SNe", 
    "SNe Only (Note: r_d is unconstrained)", 
    use_sn=True, use_cc=False, use_bao=False
)

# 2. CC Only
run_analysis(
    "Run_2_CC", 
    "CC Only (Note: r_d is unconstrained)", 
    use_sn=False, use_cc=True, use_bao=False
)

# 3. BAO Only
run_analysis(
    "Run_3_BAO", 
    "BAO Only", 
    use_sn=False, use_cc=False, use_bao=True
)


# # Run Phase 2 (2-way Joint Analyses)

# In[ ]:


# 4. SNe + CC
run_analysis(
    "Run_4_SNe_CC", 
    "SNe + CC\nOverlapping constraints: [$\Omega_m, H_0, w_0, w_a$]", 
    use_sn=True, use_cc=True, use_bao=False
)

# 5. SNe + BAO
run_analysis(
    "Run_5_SNe_BAO", 
    "SNe + BAO\nOverlapping constraints: [$\Omega_m, H_0, w_0, w_a$]", 
    use_sn=True, use_cc=False, use_bao=True
)

# 6. CC + BAO
run_analysis(
    "Run_6_CC_BAO", 
    "CC + BAO\nOverlapping constraints: [$\Omega_m, H_0, w_0, w_a$]", 
    use_sn=False, use_cc=True, use_bao=True
)


# # Run Phase 3 (Full 3-way Analysis)

# In[ ]:


# 7. SNe + CC + BAO
run_analysis(
    "Run_7_All", 
    "SNe + CC + BAO\nFully Combined Joint Inference", 
    use_sn=True, use_cc=True, use_bao=True
)


# In[ ]:





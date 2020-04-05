from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.animation as animation
import progressbar
import sys


def init_lattice(n):
    '''Create a nxn lattice with random spin configuration'''
    lattice = np.random.choice([1, -1], size=(n, n))
    return lattice

def hamiltonian(lattice, J, H):
    '''Hamiltonian of spin configuration on torus with parameters J, H, and T'''
    # J = interaction strength
    # H = external field
    # Hamiltonian : Ham = -J\sum_{i,j adj} a_i * a_j - H \sum_{i} a_i
    [n, n] = np.shape(lattice)
    ham = 0
    for i in np.arange(n):
        for j in np.arange(n):
            Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
             lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]
            ham = ham + lattice[i, j]*(-J*Sn - H)

    return ham


def deltaE(S0, Sn, J, H):
    '''Energy difference for a spin flip'''
    # S0 = spin value being flipped
    # sum of spins neighboring S0
    # J = interaction strength
    # H = external field
    # Hamiltonian : Ham = -J\sum_{i,j adj} a_i * a_j - H \sum_{i} a_i
    # flip spin a_i to -a_i
    # Energy difference = 2a_i * (H + J\sum_{j~i} a_j)
    return 2 * S0 * (H + J * Sn)


def ising(n=200,
          nsteps=500000,
          H=0,
          J=1,
          T=1,
          count_spins=False,
          countij=[1, 1],
          correlation=False,
          corr_ij=[0, 0],
          corr_r=1):
    '''Ising Model Simulator. If count_spins = True, only flipping behavior of 1 site is studied.'''

    lattice = init_lattice(n)
    energy = 0
    energies = []
    spins = []
    spin = np.sum(lattice)
    icount, jcount = countij
    counted_spins = [lattice[icount, jcount]]
    counted_intervals = []
    icorr, jcorr = corr_ij
    Sis = []
    SiSjs = []

    for step in np.arange(nsteps):

        i = np.random.randint(n)
        j = np.random.randint(n)

        # Periodic Boundary Condition
        Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
             lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]

        dE = deltaE(lattice[i, j], Sn, J, H)

        if dE < 0 or np.random.random() < np.exp(-dE / T):
            lattice[i, j] = -lattice[i, j]
            energy += dE
            energies.append(energy)
            # Note that the spin is collected at every step
            spin += 2 * lattice[i, j]

        if count_spins:
            ispin = lattice[icount, jcount]
            if ispin != counted_spins[-1]:
                counted_spins.append(ispin)
                counted_interval = step - sum(counted_intervals)

                counted_intervals.append(counted_interval)
        if correlation:
            Sn_corr = lattice[(icorr - corr_r) % n, jcorr] + lattice[(icorr + corr_r) % n, jcorr] + \
                      lattice[icorr, (jcorr - corr_r) % n] + lattice[icorr, (jcorr + corr_r) % n]
            Si = lattice[icorr, jcorr]
            SiSj = Si * Sn_corr / 4.0
            Sis.append(Si)
            SiSjs.append(SiSj)

        spins.append(spin)

    if correlation:
        return Sis, SiSjs

    if count_spins:
        return counted_spins, counted_intervals

    return lattice, energies, spins


def ising_update(X, nsteps=100, J=1, H=0, T=0.5):
    # single site Metropolis update for Ising model on n by n lattice
    # X = given lattice spin configuration
    # J = interaction strength
    # H = external field
    # T = temperature
    # Hamiltonian : Ham = -J\sum_{i,j adj} a_i * a_j - H \sum_{i} a_i
    lattice = X
    n = X.shape[0]
    energy = 0
    energies = []
    spins = []
    spin = np.sum(lattice)

    bar = progressbar.ProgressBar()
    for step in bar(range(nsteps)):
    # for step in np.arange(nsteps):
        i = np.random.randint(n)
        j = np.random.randint(n)

        # sum of spins neighboring (i,j)
        # Use periodic Boundary Condition
        Sn = lattice[(i - 1) % n, j] + lattice[(i + 1) % n, j] + \
             lattice[i, (j - 1) % n] + lattice[i, (j + 1) % n]

        # Energy difference induced by flipping spin at site (i,j)
        dE = deltaE(lattice[i, j], Sn, J, H)

        if dE < 0 or np.random.random() < np.exp(-dE / T):
            lattice[i, j] = -lattice[i, j]
            energy += dE
            energies.append(energy)
            # Note that the spin is collected at every step
            spin += 2 * lattice[i, j]

        spins.append(spin)

    return lattice, energies, spins

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 06:11:12 2023

@author: urvi
"""
# %%

import numpy as np
import matplotlib.pyplot as plt
from Project_Task4 import min_point

# %%


def rk4(upper, N, func, mew, sigma):
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]

    xi = x[:-1]
    k1 = func(xi, mew, sigma)
    k2 = func(xi + h/2, mew, sigma)
    k3 = func(xi + h/2, mew, sigma)
    k4 = func(xi + h, mew, sigma)

    integral = np.sum((k1 + 2*k2 + 2*k3 + k4) * h / 6)

    return integral, h


def gaussian(x, mew, sigma):
    return (1/(np.sqrt(2*np.pi))*np.exp(-0.5*((x)**2)))


def NB(ml, mu):  # values
    return -20 * A * np.exp(mH/k) * (np.exp(-mu/k) - np.exp(-ml/k))


def NH(ml, mu, mH, factor, sigma):  # values
    N1 = (mu)/(h) + 1
    N2 = (ml)/(h) + 1
    I1, _ = rk4(mu, int(N1), H, mH, sigma)
    I2, _ = rk4(ml, int(N2), H, mH, sigma)
    return factor*(I1 - I2)


def H(m, mew, sigma):
    return (1/sigma)*((np.sqrt(2*np.pi))**(((1/sigma)**2) - 1)) * \
        gaussian(m - mew, mew, sigma)**((1/sigma)**2)

# %% Task 5 - Calculating Probability


A = 1500
mH = 125.1
k = 20
sigma = 1.4
h = 7e-4  # ideal step size
ml = min_point[0]
mu = min_point[1]
nb = (NB(ml, mu))
nh = (NH(ml, mu, mH, factor=470, sigma=1.4))


def function(x, mu, sigma):
    g = (1/(sigma*np.sqrt(2*np.pi)))*np.e**(-0.5*((x-mu)/(sigma))**2)
    return g


upper = (5*np.sqrt(nh+nb) + nb)
#lower = 0
I1, h1 = rk4(upper, (upper/h)+1, function, nh+nb, np.sqrt(nh+nb))

prob = I1
print("Probability: ", 1 - prob)


# %% Creating the subplots for part 6 - To generate Fig 5. in my report


# NH for different higgs masses - part 6(i)
mH = 125.1
higgs_mass = np.linspace(mH - 0.2, mH + 0.2, 111)
nh_higg = np.zeros(len(higgs_mass))

for i in range(len(higgs_mass)):
    nh_higg[i] = NH(ml, mu, mH=higgs_mass[i], factor=470, sigma=1.4)

nh_max = np.max(nh_higg)
nh_mean = NH(ml, mu, mH, 470, 1.4)

# NH for different factor - part 6(iii)
nhval = 470
factorvals = np.linspace(0.97*nhval, 1.03*nhval, 111)
factorednh = np.zeros(len(factorvals))
for i in range(len(factorvals)):
    factorednh[i] = NH(ml, mu, mH=125.1, factor=factorvals[i], sigma=1.4)

factorednh_mean = NH(ml, mu, mH, 470, 1.4)

# NH for different %s - part 6(ii)
nb = NB(ml, mu)
percentages = np.linspace(0, 0.04, 111)
nh_percentages = np.zeros(len(percentages))
for i in range(len(percentages)):
    nh_percentages[i] = percentages[i] * NH(ml, mu, mH=124.5, factor=470, sigma=2.6) + (
        1-percentages[i])*NH(ml, mu, mH=125.1, factor=470, sigma=1.4)


fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 11))


ax1.plot(higgs_mass, nh_higg, '.')
ax1.axhline(nh_mean, ls='--', color='black')
ax1.fill_between(higgs_mass, nh_higg, nh_mean, where=(
    nh_higg < nh_mean), color='red', alpha=0.3)
ax1.fill_between(higgs_mass, nh_higg, nh_mean, where=(
    nh_higg >= nh_mean), color='green', alpha=0.3)

ax1.set_xlabel('Higgs Mass $(GeV/c^2)$', size=20)
ax1.set_ylabel('$N_H$', size=20)
ax1.tick_params(axis='both', which='both', labelsize=20)
ax1.set_xticks(np.linspace(125.1-0.2, 125.1+0.2, 5))
ax1.set_yticks(np.linspace(390, 394, 6))
ax1.set_title('(i)', size=20)


ax2.plot(percentages, nh_percentages, '.')
ax2.fill_between(percentages, nh_percentages, np.max(nh_percentages), where=(
    nh_percentages <= np.max(nh_percentages)), color='red', alpha=0.3)
ax2.set_ylabel('$N_H$', size=20)
ax2.set_title('(ii)', size=20)
ax2.set_xlabel('Fraction of $N_H$ affected', size=20)
ax2.tick_params(axis='both', which='both', labelsize=20)
ax2.set_xticks(np.linspace(0.00, 0.04, 6))
ax2.set_yticks(np.linspace(388, 394, 6))
ax2.set_title('(ii)', size=20)


ax3.plot(factorvals, factorednh, '.')
ax3.axhline(factorednh_mean, ls='--', color='black')
ax3.fill_between(factorvals, factorednh, factorednh_mean, where=(
    factorednh < factorednh_mean), color='red', alpha=0.3)
ax3.fill_between(factorvals, factorednh, factorednh_mean, where=(
    factorednh >= factorednh_mean), color='green', alpha=0.3)
ax3.set_ylabel('$N_H$', size=20)
ax3.set_xlabel('Number of Higgs Created', size=20)
ax3.tick_params(axis='both', which='both', labelsize=20)
ax3.set_yticks(np.linspace(380.0, 410.0, 6))
ax3.set_title('(iii)', size=20)


plt.tight_layout()
plt.subplots_adjust(wspace=None, hspace=None)

plt.show()

# %% Statistical error

nb = (NB(ml, mu))
nh = (NH(ml, mu, mH, factor=470, sigma=1.4))
stat_err = np.sqrt(nb+nh)
print("Stat error: ", stat_err)

# %% Individual positive and negative errors
err1_pos = nh_max - nh_mean
err1_neg = nh_mean - nh_higg[0]

err2_pos = 0
err2_neg = (np.max(nh_percentages[0]) - np.min(nh_percentages[-1]))

err3_pos = np.max(factorednh) - factorednh_mean
err3_neg = err3_pos

# %% Prints the combined errors

poserr = np.sqrt(err1_pos**2 + err2_pos**2 + err3_pos**2)
negerr = np.sqrt(err1_neg**2 + err2_neg**2 + err3_neg**2)
print("Statistical err: ", stat_err)
print("Combined Positive err: ", poserr)
print("Comined Negative err: ", negerr)

avgerr = (poserr + negerr)/2
print("Average combined systematic error: ", avgerr)
sig = np.sqrt(avgerr**2 + stat_err**2)
print("Total combined error: ", sig)
# %%
ml = min_point[0]
mu = min_point[1]
nb = (NB(ml, mu))
nh = (NH(ml, mu, mH, factor=470, sigma=1.4))
mu = nh+nb


def function(x, mu, sigma):
    g = (1/(sigma*np.sqrt(2*np.pi)))*np.e**(-0.5*((x-mu)/(sigma))**2)
    return g


upper = (5*sig + nb)
# lower = 1
I1, h1 = rk4(upper, (upper/h), function, mu, sig)
#I2, h2 = rk4(lower, (lower/h), function, mu, sig)

prob = I1
print("New Probability with uncertainties taken into account: ", 1 - prob)

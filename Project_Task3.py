#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 04:58:35 2023

@author: urvi
"""

# Integration of a Gaussian with mu = 0 and sigma = 1

# %% Importing necessary packages

import numpy as np
import matplotlib.pyplot as plt
from scipy import special


# %% Run Parameters

upper = 5
lower = 0

mew = 0
sigma = 1
N = np.logspace(np.log10(int(1e1)), np.log10(int(1e5)), 50)

# %% Functions to integrate and different numerical integration methods to test


def gaussian(x, mew, sigma):
    return (1/(np.sqrt(2*np.pi))*np.exp(-0.5*((x)**2)))


def newgauss(x, mew, sigma):
    if x == 0:
        g = 0
    else:
        g = (1/x**2)*gaussian(1/x, mew, sigma)
    return g


def true_erfvalue(x):
    return 0.5*special.erf(x/np.sqrt(2))


def trapezoid(upper, N, func, mew, sigma):
    result = 0
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]
    for i in range(len(x)):
        if i == 0 or i == len(x) - 1:
            result += 0.5 * func(x[i], mew, sigma)
        else:
            result += func(x[i], mew, sigma)

    result = h * result
    return result, h


def simpsonsRule(upper, N, func, mew, sigma):
    factor = 0
    result = 0
    h = (upper)/(N-1)
    x = np.linspace(lower, upper, int(N))
    h = x[1] - x[0]
    for i in range(len(x)):
        if i == 0 or i == len(x)-1:
            factor = 1/3
            result += factor*func(x[i], mew, sigma)
        elif i % 2 == 0:
            factor = 2/3
            result += factor*func(x[i], mew, sigma)
        else:
            factor = 4/3
            result += factor*func(x[i], mew, sigma)

    result = h*result
    return result, h


def euler(upper, N, f, mew, sigma):
    # h = (upper - lower)/(N+1)
    x = np.linspace(lower, upper, int(N))
    h = x[1] - x[0]
    integral = 0
    for i in range(len(x)-1):
        integral += f(x[i], mew, sigma)

    return integral*h, h


def monte(upper, N, func, mew, sigma):
    V = upper
    I = 0
    x = upper * np.random.rand(int(N)+1)
    for i in range(1, (int(N) + 1)):
        I += func(x[i], mew, sigma)

    return (V * I/(int(N)+1)), N


def rk2(upper, N, func, mew, sigma):
    alpha = 0.5
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]

    integral = 0
    for i in range(len(x)-1):
        k1 = func(x[i], mew, sigma)
        k2 = func(x[i] + alpha*h, mew, sigma)
        integral += (1/(2*alpha))*((2*alpha - 1)*k1 + k2)

    return (h) * integral, h


def rk4(upper, N, func, mew, sigma):
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]

    integral = 0
    for i in range(len(x)-1):
        k1 = func(x[i], mew, sigma)
        k23 = 2*func(x[i] + h/2, mew, sigma)
        k4 = func(x[i] + h, mew, sigma)

        integral += (k1 + 2*k23 + k4)

    return (h/6) * integral, h


# Derivatives of gaussian
def df_dx(x, func, mew, sigma):
    return -x*func(x, mew, sigma)


def df2_dx(x, func, mew, sigma):
    return (-1 + x**2)*func(x, mew, sigma)


def df3_dx(x, func, mew, sigma):
    return (3*x - x**3)*func(x, mew, sigma)


# Derivatives of newgauss
def newdf_dx(x, func, mew, sigma):
    if x == 0:
        return 0
    else:
        return ((1-(2*x**2))/x**3)*func(x, mew, sigma)


def newdf2_dx(x, func, mew, sigma):
    if x == 0:
        return 0
    else:
        return (((6*x**4)-(7*x**2)+1)/x**6) * func(x, mew, sigma)


def newdf3_dx(x, func, mew, sigma):
    if x == 0:
        return 0
    else:
        return (((-24*x**6)+(48*x**4)-(15*x**2)+1)/x**9) * func(x, mew, sigma)


def HOT2(upper, N, func, mew, sigma, dfdx):
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]
    integral = 0
    T2 = 0
    for i in range(len(x) - 1):
        T2 = func(x[i], mew, sigma) + (h/2)*dfdx(x[i], func, mew, sigma)
        integral += h*T2
    return integral, h


def HOT4(upper, N, func, mew, sigma, dfdx, df2dx, df3dx):
    x = np.linspace(0, upper, int(N))
    h = x[1] - x[0]
    integral = 0
    T4 = 0
    for i in range(len(x)-1):
        T4 = func(x[i], mew, sigma) + (h/2) * dfdx(x[i], func, mew, sigma) + ((h**2)/6) * \
            df2dx(x[i], func, mew, sigma) + ((h**3)/24) * \
            df3dx(x[i], func, mew, sigma)
        integral += h*T4
    return integral, h


'''
Online research suggested that gauss_quadrature was the most accurate method of
integration so I tried to implement it. However, it did not run (took too long) 
so I was not able to compare it with the other methods.
'''

# def gauss_quadrature(func, a, b, n, mew, sigma):
#     # Get the nodes (x) and weights (w) for Gauss quadrature
#     x, w = np.polynomial.legendre.leggauss(int(n))
#     # Map the nodes from [-1, 1] to [a, b]
#     x_mapped = 0.5 * (b - a) * x + 0.5 * (b + a)
#     h = x_mapped[1] - x_mapped[0]

#     # Vectorized computation of the integral
#     integral = np.sum(w * func(x_mapped, mew, sigma))

#     # Scale by half the interval length
#     integral *= 0.5 * (b - a)

#     return integral, h

# %% Calculating errors from each of the methods

true = true_erfvalue(upper) - true_erfvalue(lower)

err1 = np.zeros(len(N))
err2 = np.zeros(len(N))
err3 = np.zeros(len(N))
err4 = np.zeros(len(N))
err5 = np.zeros(len(N))
err6 = np.zeros(len(N))
err7 = np.zeros(len(N))
err8 = np.zeros(len(N))
err9 = np.zeros(len(N))
err10 = np.zeros(len(N))
err11 = np.zeros(len(N))
err12 = np.zeros(len(N))
err13 = np.zeros(len(N))
err14 = np.zeros(len(N))
err15 = np.zeros(len(N))
err16 = np.zeros(len(N))


h1 = np.zeros(len(N))
h2 = np.zeros(len(N))
h3 = np.zeros(len(N))
h4 = np.zeros(len(N))
h5 = np.zeros(len(N))
h6 = np.zeros(len(N))
h7 = np.zeros(len(N))
h8 = np.zeros(len(N))
h9 = np.zeros(len(N))
h10 = np.zeros(len(N))
h11 = np.zeros(len(N))
h12 = np.zeros(len(N))
h13 = np.zeros(len(N))
h14 = np.zeros(len(N))
N15 = np.zeros(len(N))
N16 = np.zeros(len(N))

for i in range(len(N)):

    result1, h1[i] = trapezoid(upper, int(N[i]), gaussian, mew, sigma)
    result2, h2[i] = trapezoid(1/upper, int(N[i]), newgauss, mew, sigma)
    result2 = 0.5 - result2

    result3, h3[i] = euler(upper, int(N[i]), gaussian, mew, sigma)
    result4, h4[i] = euler(1/upper, int(N[i]), newgauss, mew, sigma)
    result4 = 0.5 - result4

    result5, h5[i] = rk2(upper, int(N[i]), gaussian, mew, sigma)
    result6, h6[i] = rk2(1/upper, int(N[i]), newgauss, mew, sigma)
    result6 = 0.5 - result6

    result7, h7[i] = rk4(upper, int(N[i]), gaussian, mew, sigma)
    result8, h8[i] = rk4(1/upper, int(N[i]), newgauss, mew, sigma)
    result8 = 0.5 - result8

    # # # HOT4
    result9, h9[i] = HOT2(upper,  int(N[i]), gaussian, mew, sigma, df_dx)
    result10, h10[i] = HOT2((1/upper), int(N[i]),
                            newgauss, mew, sigma, newdf_dx)
    result10 = 0.5 - result10

    result11, h11[i] = HOT4(upper,  int(N[i]), gaussian,
                            mew, sigma, df_dx, df2_dx, df3_dx)
    result12, h12[i] = HOT4((1/upper), int(N[i]), newgauss,
                            mew, sigma, newdf_dx, newdf2_dx, newdf3_dx)
    result12 = 0.5 - result12

    result15, N15[i] = monte(upper,  int(N[i]), gaussian, mew, sigma)
    result16, N16[i] = monte((1/upper), int(N[i]),
                             newgauss, mew, sigma)
    result16 = 0.5 - result16

    err1[i] = (abs(result1 - true)/true)
    err2[i] = (abs(result2 - true)/true)
    err3[i] = (abs(result3 - true)/true)
    err4[i] = (abs(result4 - true)/true)
    err5[i] = (abs(result5 - true)/true)
    err6[i] = (abs(result6 - true)/true)
    err7[i] = (abs(result7 - true)/true)
    err8[i] = (abs(result8 - true)/true)
    err9[i] = (abs(result9 - true)/true)
    err10[i] = (abs(result10 - true)/true)
    err11[i] = (abs(result11 - true)/true)
    err12[i] = (abs(result12 - true)/true)
    err15[i] = (abs(result15 - true)/true)
    err16[i] = (abs(result16 - true)/true)


# %%
'''
Simpson's Rule errors are calculated in a seperate loop as we need to ensure
that N is even just for this method'
'''

for i in range(len(N)):

    if int(N[i]) % 2 == 0:
        N[i] += 1
    else:
        pass

    result13, h13[i] = simpsonsRule(upper, int(N[i]), gaussian, mew, sigma)

    err13[i] = (abs(result13 - true)/true)

    result14, h14[i] = simpsonsRule(1/upper, int(N[i]), newgauss, mew, sigma)
    result14 = 0.5 - result14
    err14[i] = (abs(result14 - true)/true)


# %% To generate Fig 1. my report


plt.plot(h3, (err3), 'o', label='Euler', markersize=5, lw=2)
plt.plot(h9, (err9), 'o', label='HOT2', markersize=5, lw=2)
plt.plot(h1, (err1), 'o', label='Trapezoid', markersize=5, lw=2)
plt.plot(h5, (err5), 'o', label='RK2', markersize=5, lw=2)
plt.plot(h11, (err11), 'o', label='HOT4', markersize=5, lw=2)
plt.plot(h13, (err13), 'o', label="Simpson's", markersize=5, lw=2)
plt.plot(h7, (err7), 'o', label='RK4', markersize=5, lw=1)

plt.legend(bbox_to_anchor=(0.9, 0.78), loc='upper left', fontsize=20)
plt.tick_params(axis='both', which='both', labelsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Stepsize (h)', size=22)
plt.ylabel('Relative error, $\\varepsilon_g$', size=22)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %% To generate Fig 2. in my report

plt.plot(h4, (err4), 'o', label='Euler', markersize=5, lw=2)
plt.plot(h10, (err10), 'o', label='HOT2', markersize=5, lw=2)
plt.plot(h2, (err2), 'o', label='Trapezoid', markersize=5, lw=2)
plt.plot(h6, (err6), 'o', label='RK2', markersize=5, lw=2)
plt.plot(h12, (err12), 'o', label='HOT4', markersize=5, lw=2)
plt.plot(h14, (err14), 'o', label="Simpson's", markersize=5, lw=2)
plt.plot(h8, (err8), 'o', label='RK4', markersize=5, lw=1)

plt.legend(bbox_to_anchor=(0.9, 0.78), loc='upper left', fontsize=20)
plt.tick_params(axis='both', which='both', labelsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Stepsize (h)', size=22)
plt.ylabel('Relative error, $\\varepsilon_g$', size=22)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %% Monte Carlo Results
'''
Though I do not include my Monte Carlo results in my report, I do mention that
it was trialled and found to be less accurate than any of the other methods. 
This cell produces graphs to back this claim.
'''
# 0 to a method
plt.plot(N15, (err15), 'o', label='Monte Carlo 0 to a method', markersize=5, lw=1)


# a to infinity method
plt.plot(N16, (err16), 'o', label='Monte Carlo a to inf method', markersize=5, lw=1)

plt.legend(bbox_to_anchor=(0.9, 0.78), loc='upper left', fontsize=20)
plt.tick_params(axis='both', which='both', labelsize=22)
plt.grid(True, linestyle='--', alpha=0.7)
plt.xlabel('Stepsize (h)', size=22)
plt.ylabel('Relative error, $\\varepsilon_g$', size=22)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
'''
Studying the effect of changing a on ideal step size. Generates Fig 3.
in my report
'''

N = np.logspace(np.log10(1e1), np.log10(1e4), 35)
upper_values_left = [0.2, 0.5, 0.8]
upper_values_right = [1, 3, 5]

fig, axs = plt.subplots(1, 2, figsize=(8, 11), sharey=True)
axs[0].set_xlabel('Stepsize (h)', size=20)
axs[0].set_ylabel('Relative error, $\\varepsilon_g$', size=20)
axs[0].set_xscale('log')
axs[0].set_yscale('log')
axs[0].tick_params(axis='both', which='both', labelsize=20)

for i, upper in enumerate(upper_values_left):
    true = true_erfvalue(upper)
    err0toa = np.zeros(len(N))
    erratoinf = np.zeros(len(N))
    h0toa = np.zeros(len(N))
    hatoinf = np.zeros(len(N))

    for j in range(len(N)):
        res0toa, h0toa[j] = rk4(
            upper, int(N[j]), gaussian, 0, 1)
        resatoinf, hatoinf[j] = rk4(
            1 / upper, int(N[j]), newgauss, 0, 1)
        resatoinf = 0.5 - resatoinf

        err0toa[j] = abs(res0toa - true) / true
        erratoinf[j] = abs(resatoinf - true) / true

    mask = h0toa < 0.05
    axs[0].plot(h0toa[mask], err0toa[mask], 'o',
                label=f'$0$ to ${upper}$', markersize=4, lw=1, color=f'C{i+1}')

    mask = hatoinf < 0.05
    axs[0].plot(hatoinf[mask], erratoinf[mask], 's',
                label=f'${upper}$ to $\infty$', markersize=4, lw=1, color=f'C{i+1}')


axs[1].set_xlabel('Stepsize (h)', size=20)
axs[1].set_xscale('log')
axs[1].set_yscale('log')
axs[1].tick_params(axis='both', which='both', labelsize=20)


for i, upper in enumerate(upper_values_right):
    true = true_erfvalue(upper)
    err0toa = np.zeros(len(N))
    erratoinf = np.zeros(len(N))
    h0toa = np.zeros(len(N))
    hatoinf = np.zeros(len(N))

    for j in range(len(N)):
        result0toa, h0toa[j] = rk4(
            upper, int(N[j]), gaussian, 0, 1)
        resultatoinf, hatoinf[j] = rk4(
            1 / upper, int(N[j]), newgauss, 0, 1)
        resultatoinf = 0.5 - resultatoinf

        err0toa[j] = abs(result0toa - true) / true
        erratoinf[j] = abs(resultatoinf - true) / true

    mask = h0toa < 0.1
    axs[1].plot(h0toa[mask], err0toa[mask], 'o',
                label=f'$0$ to ${upper}$', markersize=4, lw=1, color=f'C{i+1}')

    mask = hatoinf < 0.1
    axs[1].plot(hatoinf[mask], erratoinf[mask], 's',
                label=f'${upper}$ to $\infty$', markersize=4, lw=1, color=f'C{i+1}')

h0 = 1e-3 - (1e-3 - 1e-4)/3
h1 = 1e-3 - (1e-3 - 1e-4)/1.15

axs[0].axvline(h0, ls='--', color='black',
               label='$ h_o \\approx 7.0 \\times 10^{-4}$')
axs[1].axvline(h1, ls='--', color='black',
               label='$h_o \\approx 2.2 \\times 10^{-4}$')
axs[0].legend(fontsize=16, loc='upper left')
axs[1].legend(fontsize=16, loc='upper left')
axs[0].grid(True, linestyle='--', alpha=0.7)
axs[1].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

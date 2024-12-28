#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 05:52:42 2023

@author: urvi
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

# %%
'''
#Note that I have re written a more efficient rk4 method for this part
'''


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


# %%
A = 1500
mH = 125.1
k = 20
sigma = 1.4

h = 7e-4  # ideal step size


def H(m, mew, sigma):
    return (1/sigma)*((np.sqrt(2*np.pi))**(((1/sigma)**2) - 1)) * \
        gaussian(m - mew, mew, sigma)**((1/sigma)**2)


def NB(ml, mu):  # values
    return -20 * A * np.exp(mH/k) * (np.exp(-mu/k) - np.exp(-ml/k))


def NH(ml, mu, mH, factor, sigma):  # values
    N1 = (mu)/(h)
    N2 = (ml)/(h)
    I1, _ = rk4(mu, int(N1), H, mH, sigma)
    I2, _ = rk4(ml, int(N2), H, mH, sigma)
    return factor*(I1 - I2)


def significance(ml, mu):
    return -NH(ml, mu, mH, factor=470, sigma=1.4)/np.sqrt(NB(ml, mu))


# %%
'''
Task 4(a) - Investigating significance with different mass cuts. Generates 
Fig. 4 in my report

'''


ml = np.linspace(mH - 5*sigma, mH, 10)
mu = np.linspace(mH+1, mH+5*sigma, 10)
sig = np.zeros((len(mu), len(ml)))

for i, mu_val in enumerate(mu):
    sig[i, :] = [-significance(ml_val, mu_val) for ml_val in ml]

plt.contourf(ml, mu, sig, levels=50)
plt.xlabel('ml (GeV/c^2)')
plt.ylabel('mu (GeV/c^2)')
plt.title('Gradient Descent to Maximise Significance')


# %%
'''
Task 4(b) Finding the ideal mass cuts that optimise significance. I tried two
methods of maximisation, gradient descent and quasi newton. I decided to use
Gradient descent as it gave mass cuts to the required accuracy with sufficient
efficiency. Quasi Newton required an extra constraint of the Hessian being 
positive definite.
'''


def grad(func, ml, mu, h=1e-8):
    grad_ml = (func(ml + h, mu) - func(ml - h, mu)) / (2 * h)
    grad_mu = (func(ml, mu + h) - func(ml, mu - h)) / (2 * h)
    return np.array([grad_ml, grad_mu])


def grad_des(func, start, alpha, tolerance=1e-7):
    current_point = np.array(start)
    gradient = grad(func, current_point[0], current_point[1])

    while(np.linalg.norm(gradient) > tolerance):
        gradient = grad(func, current_point[0], current_point[1])
        current_point -= alpha*gradient

    return current_point, func(*current_point)


# def quasi_newton(func, start, alpha, tolerance=1e-7):
#     current_point = np.array(start)
#     gradient = grad(func, current_point[0], current_point[1])
#     G = np.identity(2)

#     while(np.linalg.norm(gradient) > tolerance):

#         new_point = current_point - alpha*np.dot(G, gradient)

#         new_gradient = grad(func, new_point[0], new_point[1])

#         delta = new_point - current_point
#         gamma = new_gradient - gradient

#         delouter = np.outer(delta, delta)
#         gamouter = np.outer(gamma, gamma)
#         G = G - (np.dot(G, np.dot(gamouter, G))) / np.dot(gamma,
#                                                           np.dot(G, gamma)) + (delouter / abs(np.dot(gamma, delta)))

#         current_point = new_point
#         gradient = new_gradient
#     return current_point, func(*current_point)


start = [mH - 2*sigma, mH + 2*sigma]


min_point, min_value = grad_des(significance, start, 1)
#maxquas, maxval = quasi_newton(significance, start, 1)

print("Grad Des Maximum Point:", min_point)
print("Grad Des Maximum Significance:", -min_value)

# print("Quasi Newton Maximum Point:", maxquas)
# print("Quasi Newton Maximum Significance:", -maxval)
# %%

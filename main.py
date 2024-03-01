import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad


# Base function
def base_function(i, n, x):
    h = 2 / n
    if h * (i - 1) <= x <= h * i:
        return (x - h * (i - 1)) / h
    elif h * i < x <= h * (i + 1):
        return (h * (i + 1) - x) / h
    else:
        return 0


# Derivative of base function
def deriv(i, n, x):
    h = 2 / n
    if h * (i - 1) <= x < h * i:
        return 1 / h
    elif h * i <= x < h * (i + 1):
        return -1 / h
    else:
        return 0


# Calculate B(u, v)
def calc_b(u, v, n):
    h = 2 / n
    u -= 1
    v -= 1
    B = -base_function(u, n, 2) * base_function(v, n, 2)

    def integrand_deriv(x):
        return deriv(u, n, x) * deriv(v, n, x)

    def integrand_base(x):
        return base_function(u, n, x) * base_function(v, n, x)

    def integrand_deriv2(x):
        return deriv(u, n, x) * deriv(u, n, x)

    def integrand_base2(x):
        return base_function(u, n, x) * base_function(u, n, x)

    # Upper diagonal
    if u == v - 1:
        a = u * h
        b = (u + 1) * h
        integral_deriv = quad(integrand_deriv, a, b)[0]
        integral_base = quad(integrand_base, a, b)[0]
        B = B + integral_deriv - integral_base

    # Lower diagonal
    elif u == v + 1:
        a = (u - 1) * h
        b = u * h
        integral_deriv = quad(integrand_deriv, a, b)[0]
        integral_base = quad(integrand_base, a, b)[0]
        B = B + integral_deriv - integral_base

    # Main diagonal
    elif u == n:
        a = (u - 1) * h
        b = u * h
        integral_deriv = quad(integrand_deriv2, a, b)[0]
        integral_base = quad(integrand_base2, a, b)[0]
        B = B + integral_deriv - integral_base
    else:
        a = (u - 1) * h
        b = (u + 1) * h
        integral_deriv = quad(integrand_deriv2, a, b)[0]
        integral_base = quad(integrand_base2, a, b)[0]
        B = B + integral_deriv - integral_base
    return B


# Calculate L(v)
def calc_l(v, n):
    v -= 1
    h = 2 / n
    a = (v - 1) * h
    b = (v + 1) * h
    return quad(lambda x: math.sin(x) * base_function(v, n, x), a, b)[0]


# Main function
def main():
    n = int(input("Enter the number of elements: "))
    h = 2 / n

    b_matrix = np.zeros((n + 1, n + 1))
    # Dirichlet boundary condition
    b_matrix[0, 0] = 1

    l_matrix = np.zeros(n + 1)

    for i in range(2, n + 2):
        b_matrix[i - 1, i - 1] = calc_b(i, i, n)
        if i < n + 1:
            b_matrix[i - 1, i] = calc_b(i, i + 1, n)
        if i > 2:
            b_matrix[i - 1, i - 2] = calc_b(i, i - 1, n)
        l_matrix[i - 1] = calc_l(i, n)

    result = np.linalg.solve(b_matrix, l_matrix)
    x = np.arange(0, 2 + h, h)
    print("Result:\n", result)
    print("Determinant of B matrix:", np.linalg.det(b_matrix))

    plt.plot(x, result)
    plt.show()


main()

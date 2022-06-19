from __future__ import annotations

from enum import Enum
from typing import List, Tuple

from math import factorial, sqrt, isinf, exp, inf
from scipy.stats import norm, chi2
from scipy.stats import t as t_student

norm_cdf = norm.cdf
norm_ppf = norm.ppf

chi2_ppf = chi2.ppf

t_ppf = t_student.ppf


class TestType(Enum):
    LEFT_SIDED = 0
    RIGHT_SIDED = 1
    DOUBLE_SIDED = 2


class WrongTestTypeError(TypeError):
    """Raised when TestType is not supplied"""
    pass


def pois_with_lambda(l: float | int, k: int) -> float:
    """
    Returns approximate probability for Poisson distribution
    with parameters k and lambda

            Parameters:
                    l (float|int): Lambda parameter
                    k (int): Number of successes

            Returns:
                    prob (float): Probability
    """
    return (l ** k) / factorial(k) * (exp(-l))


def pois(n: int, p: float, k: int) -> float:
    """
    Returns approximate probability (for Poisson distribution)
    of event of probability p happening exactly k times in n tries

            Parameters:
                    n (int): Number of tries
                    p (float): Probability of success
                    k (int): Number of successes

            Returns:
                    prob (float): Probability
    """

    lam = n * p
    return pois_with_lambda(lam, k)


def pois2_with_lambda(l: float | int, k: List[int]) -> float:
    """
    Returns approximate probability (for Poisson distribution)
    for given lambda and list of k parameters

            Parameters:
                    l (float|int): Lambda parameter
                    k (List[int]): Another decimal integer

            Returns:
                    v (float): Probability
    """

    prob = sum([pois_with_lambda(l, v) for v in k])

    return prob


def pois2(n: int, p: float, k: List[int]) -> Tuple[float, float]:
    """
    Returns approximate probability (for Poisson distribution)
    of event of probability p happening all of k times in n tries

            Parameters:
                    n (int): Number of tries
                    p (float): Probability of success
                    k (List[int]): Another decimal integer

            Returns:
                    v (float, float): Probability and maximal error
    """

    l = n * p
    prob = sum([pois(n, p, v) for v in k])

    err = (l ** 2) / n

    return prob, err


def norm(n, m, s, t) -> float:
    """
    Calculates approximate probability using normal distribution

            Parameters:
                    n (int): Number of variables
                    m (float): Expected value
                    s (float): Standard deviation
                    t (float): Upper limit

            Returns:
                    prob (float): Approximate probability
    """
    v = (t - (n * m)) / (s * sqrt(n))
    return norm_cdf(v)


def norm2(n, m, s, d, g):
    """
    Calculates approximate probability using normal distribution given
    upper and lower limit

            Parameters:
                    n (int): Number of variables
                    m (float): Expected value
                    s (float): Standard deviation
                    d (float): Lower limit
                    g (float): Upper limit

            Returns:
                    prob (float): Approximate probability
    """

    if isinf(d):
        return norm(n, m, s, g)

    if isinf(g):
        return 1 - norm(n, m, s, d)

    return norm(n, s, s, g) - norm(n, m, s, d)


def mean(l: List[int | float]) -> float:
    """
    Calculates mean

            Parameters:
                    l (List[int|float]): List of numbers to calculate mean from

            Returns:
                    mean (float): Mean
    """
    return sum(l) / len(l)


def new(l: List[int | float]) -> float:
    """
    Calculates unbiased estimator of variance

            Parameters:
                    l (List[int|float]): List of numbers to calculate UEV from

            Returns:
                    new (float): Mean
    """
    m = mean(l)
    s = 0

    for el in l:
        s += (el - m) ** 2

    return s / (len(l) - 1)


def estimate_mean_with_variation(l: List[float | int], sd: float, alpha: float) -> Tuple[float, float]:
    """
    Estimates mean with known variation, returns confidence interval
    for designated confidence level

            Parameters:
                    l (List[float|int]): List of samples
                    sd (float): Standard deviation
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                    ci ((float, float)): Confidence interval
    """
    m = mean(l)

    bound = norm_ppf(1 - (alpha / 2)) * sd / sqrt(len(l))

    return m - bound, m + bound


def estimate_mean(l, alpha):
    """
    Estimates mean with unknown variation for designated confidence level

            Parameters:
                    l (List[float|int]): List of samples
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                    ci ((float, float)): Confidence interval
    """
    n = len(l)
    m = mean(l)
    bound = t_ppf(1 - (alpha / 2), n - 1) * sqrt(new(l)) / sqrt(n)

    return m - bound, m + bound


def estimate_variation(l: List[float | int], alpha: float) -> Tuple[float, float]:
    """
    Estimates variation for designated confidence level

            Parameters:
                    l (List[float|int]): List of samples
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                    ci ((float, float)): Confidence interval
    """
    n = len(l)
    lower = (n - 1) * new(l) / chi2_ppf(1 - (alpha / 2), n - 1)
    upper = (n - 1) * new(l) / chi2_ppf((alpha / 2), n - 1)

    return lower, upper


def estimate_proportion(l: List[str], s: str, alpha: float) -> Tuple[float, float]:
    """
    Estimates proportion for designated confidence level

            Parameters:
                    l (List[str]): List of samples (words)
                    s (str): Word to calculate proportion for
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                    ci ((float, float)): Confidence interval
    """
    p = 0
    n = len(l)

    for el in l:
        if el == s:
            p += 1

    p = p / n

    bound = norm_ppf(1 - (alpha / 2)) * sqrt(p * (1 - p) / n)

    return p - bound, p + bound


def check_mean(l: List[float | int], hyp: int | float, sd: int | float, alpha: float, t: TestType) -> bool:
    """
    Checks if mean belongs to given distribution for designated confidence level
    and test type

            Parameters:
                    l (List[float|int]): List of samples which hyp can come from
                    hyp (int|float): Hypothetical mean to check
                    sd (int|float): Standard deviation of the sample, -1 if not available
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)
                    t (TestType): Test type to check

            Returns:
                    result (bool): Whether hyp comes from sample or not

            Raises:
                    WrongTestTypeError: t parameter is of wrong type
    """
    n = len(l)
    # ppf = None

    if sd == -1:
        sd = sqrt(new(l))

    w = (mean(l) - hyp) / (sd / sqrt(n))

    if n >= 30:
        ppf = norm_ppf
    else:
        ppf = lambda p: t_ppf(p, n - 1)

    if t == TestType.RIGHT_SIDED:
        return w > ppf(1 - alpha)

    if t == TestType.LEFT_SIDED:
        return w < -ppf(1 - alpha)

    if t == TestType.DOUBLE_SIDED:
        return ppf(1 - (alpha / 2)) < w or w < -ppf(1 - (alpha / 2))

    raise WrongTestTypeError


def compare_mean(l1, l2, sd1, sd2, alpha, t: TestType) -> bool:
    """
    Checks if means belongs to the same distribution

            Parameters:
                    l1 (List[float|int]): List of samples no. 1
                    l2 (List[float|int]): List of samples no. 2
                    sd1 (int|float): Standard deviation of sample 1
                    sd2 (int|float): Standard deviation of sample 2
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)
                    t (TestType): Test type to check

            Returns:
                    result (bool): Whether hyp comes from sample or not

            Raises:
                    WrongTestTypeError: t parameter is of wrong type
    """
    n1 = len(l1)
    n2 = len(l2)
    n = n1 + n2

    # w = None
    # ppf = None

    if sd1 < 0 or sd2 < 0:
        s = (n1 * new(l1) + n2 * new(l2)) / (n1 + n2 - 2)
        w = (mean(l1) - mean(l2)) / (sqrt(s) * sqrt(1 / n1 + 1 / n2))
    else:
        w = (mean(l1) - mean(l2)) / sqrt((sd1 ** 2) / n1 + (sd2 ** 2) / n2)

    if n >= 30:
        ppf = norm_ppf
    else:
        ppf = lambda p: t_ppf(p, n - 1)

    if t == TestType.RIGHT_SIDED:
        return w > ppf(1 - alpha)

    if t == TestType.LEFT_SIDED:
        return w < -ppf(1 - alpha)

    if t == TestType.DOUBLE_SIDED:
        return ppf(1 - (alpha / 2)) < w or w < -ppf(1 - (alpha / 2))

    raise WrongTestTypeError


def correlation(l1, l2, hyp, alpha, t: TestType) -> bool:
    """
    Checks pair by pair if two vectors of variables are correlated

            Parameters:
                    l1 (List[float|int]): List of samples no. 1
                    l2 (List[float|int]): List of samples no. 2
                    hyp (int|float): Hypothetical mean
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)
                    t (TestType): Test type to check

            Returns:
                    result (bool): True if correlated, false otherwise

            Raises:
                    WrongTestTypeError: t parameter is of wrong type
    """
    if len(l1) != len(l2):
        raise "len(l1) != len(l2)"

    d = [v1 - v2 for (v1, v2) in zip(l1, l2)]
    n = len(d)

    w = (mean(d) - hyp) / (sqrt(new(d)) / sqrt(n))

    if t == TestType.RIGHT_SIDED:
        return w > t_ppf(1 - alpha, n - 1)

    if t == TestType.LEFT_SIDED:
        return w < -t_ppf(1 - alpha, n - 1)

    if t == TestType.DOUBLE_SIDED:
        return t_ppf(1 - (alpha / 2), n - 1) < w or w < -t_ppf(1 - (alpha / 2), n - 1)

    raise WrongTestTypeError


def check_variance(l, hyp, alpha, t: TestType) -> bool:
    """
    Checks if variance comes from the sample

            Parameters:
                    l (List[float|int]): List of samples
                    hyp (int|float): Hypothetical variance
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)
                    t (TestType): Test type to check

            Returns:
                    result (bool): True if hyp comes from sample, false otherwise

            Raises:
                    WrongTestTypeError: t parameter is of wrong type

    """

    n = len(l)

    w = (n - 1) * new(l) / hyp

    if t == TestType.LEFT_SIDED:
        return w < chi2_ppf(alpha, n - 1)

    if t == TestType.RIGHT_SIDED:
        return w > chi2_ppf(1 - alpha, n - 1)

    if t == TestType.DOUBLE_SIDED:
        return chi2_ppf(1 - (alpha / 2), n - 1) < w or w < chi2_ppf(alpha / 2, n - 1)

    raise WrongTestTypeError


def check_uniform(l, alpha) -> bool:
    """
    Checks if l comes from uniform distribution

            Parameters:
                    l (List[float|int]): List of samples
                    alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                    result (bool): True if l comes from uniform distribution
    """
    m = mean(l)
    n = len(l)

    w = sum([ ((v - m) ** 2) / m for v in l])

    return w > chi2_ppf(1 - alpha, n - 1)


def check_pois(l, lam, alpha) -> bool:
    """
    Checks if l comes from Poisson distribution

            Parameters:
                l (List[float|int]): List of samples
                lam (float): Lambda parameter for Poisson distribution
                alpha (float): Statistical significance (where 1-alpha equals confidence level)

            Returns:
                result (bool): True if l comes from Poisson distribution
    """
    w = 0
    n = len(l)

    for i, v in enumerate(l):
        w += ((v - pois_with_lambda(lam, i)) ** 2) / pois_with_lambda(lam, i)

    return w > chi2_ppf(1 - alpha, n - 2)


print("-" * 13)
print("| Zestaw 10 |")
print("-" * 13)

print("Zadanie 1")
s1ex1_n = 100000
s1ex1_p_error = (1 / 1000) * (1 / 10) * (1 / 2)
s1ex1_p, s1ex1_err = pois2(s1ex1_n, s1ex1_p_error, [0, 1, 2, 3])
print(s1ex1_p)
print()

print("Zadanie 2")
s1ex2_lambda = 0
while 1-pois_with_lambda(s1ex2_lambda, 0) < 0.99:
    s1ex2_lambda += 1
print(s1ex2_lambda)
print()

print("Zadanie 3")
s1ex3_lambda = 1
s1ex3_k = 0
s1ex3_p = pois_with_lambda(s1ex3_lambda, s1ex3_k)
print(s1ex3_p)
print()

print("Zadanie 4")
print("WIP")
print()

print("Zadanie 5")
s1ex5_n = 1000
s1ex5_ex = (1+2+3+4+5+6)/6
s1ex5_ex2 = (1+4+9+16+25+36)/6
s1ex5_sd = sqrt(s1ex5_ex2 - (s1ex5_ex**2))
s1ex5_p = norm2(1000, s1ex5_ex, s1ex5_sd, 3450, 3550)
print(s1ex5_p)

s1ex5_m = s1ex5_n * s1ex5_ex
s1ex5_delta = 0
while norm2(1000, s1ex5_ex, s1ex5_sd, s1ex5_m-s1ex5_delta, s1ex5_m+s1ex5_delta) < 0.99:
    s1ex5_delta += 1
print(s1ex5_m-s1ex5_delta, s1ex5_m+s1ex5_delta)
print()

print("Zadanie 6")
ex6_cash = 0
while norm2(400, 0.0, 100.0, -ex6_cash, inf) < 0.99:
    ex6_cash += 1
print(ex6_cash)
print()

print("Zadanie 11")
s1ex11_lambda = 90
s1ex11_l = list(range(80, 110+1))
s1ex11_p = pois2_with_lambda(s1ex11_lambda, s1ex11_l)
print(s1ex11_p)
print()


print()
print("-" * 13)
print("| Zestaw 11 |")
print("-" * 13)

print("Zadanie 1")
print(estimate_mean_with_variation([680]*81, 270, 1-0.95))
print()

print("Zadanie 2")
print(f'95% ufności: {estimate_mean_with_variation([139000]*36, 12000, 1-0.95)}')
print(f'99% ufności: {estimate_mean_with_variation([139000]*36, 12000, 1-0.99)}')
print()

print("Zadanie 3")
s2ex3_l = [8.1*1000, 7.9*1000, 9.6*1000, 6.4*1000, 8.7*1000, 8.8*1000, 7.9*1000]
print(estimate_mean(s2ex3_l, 1-0.90))
print()

print("Zadanie 4")
s2ex4_l = [8, 12, 26, 10, 23, 21, 16, 22, 18, 17, 36, 9]
print(estimate_mean(s2ex4_l, 1-0.99))
print()

print("Zadanie 6")
s2ex6_l = ["tak"] * 48 + ["nie"] * (169-48)
print(estimate_proportion(s2ex6_l, "tak", 1-0.90))
print()


print()
print("-" * 13)
print("| Zestaw 12 |")
print("-" * 13)


print("Zadanie 1")
s3ex1_l = [1] * 18 + [0] * (900-18)
s3ex1_alpha = 0.05
s3ex1_hyp = 0.03
s3ex1_p = check_mean(s3ex1_l, s3ex1_hyp, -1, s3ex1_alpha, TestType.LEFT_SIDED)
print("H0 – butelki wybrakowane >= 0.03")
print("Odrzucamy H0" if s3ex1_p else "Nie ma podstaw by odrzucić H0")
print()

print("Zadanie 2")
s3ex2_l = [1] * 3 + [0] * (15-3)
s3ex2_alpha = 0.1
s3ex2_hyp = 0.06
s3ex2_p = check_mean(s3ex2_l, s3ex2_hyp, -1, s3ex2_alpha, TestType.LEFT_SIDED)
print("H0 – niespełniające wymagań >= 0.06")
print("Odrzucamy H0" if s3ex2_p else "Nie ma podstaw by odrzucić H0")
print()

print("Zadanie 3")
s3ex3_l1 = [1] * 11 + [0] * (20-11)
s3ex3_l2 = [1] * 17 + [0] * (20-17)
s3ex3_alpha = 0.01
s3ex3_p = compare_mean(s3ex3_l1, s3ex3_l2, -1, -1, s3ex3_alpha, TestType.LEFT_SIDED)
print("H0 – skuteczność A >= skuteczność B")
print("Odrzucamy H0" if s3ex3_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 4")
s3ex4_l = [8.235, 8.183, 8.207, 8.156, 8.167, 8.199, 8.122, 8.186, 8.169, 8.199,
           8.245, 8.181, 8.204, 8.219, 8.183, 8.264, 8.205, 8.181, 8.186, 8.224]
s3ex4_alpha = 0.01
s3ex4_sigma = 0.03
s3ex4_hyp = 8.1
s3ex4_p = check_mean(s3ex4_l, s3ex4_hyp, s3ex4_sigma, s3ex4_alpha, TestType.RIGHT_SIDED)
print("H0 – średnica <= 8.1")
print("Odrzucamy H0" if s3ex4_p else "Nie ma podstaw by odrzucić H0")
print()

print("Zadanie 8")
s3ex8_l = [16, 17, 19, 16, 24, 19, 17, 16]
s3ex8_alpha = 0.05
s3ex8_p = check_uniform(s3ex8_l, s3ex8_alpha)
print("H0 – próbka pochodzi z rozkładu jednostajnego")
print("Odrzucamy H0" if s3ex8_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 11")
# Toshiba -> 1  inne -> 0
s3ex11_l1 = [1]*285 + [0]*(1000-285)
s3ex11_l2 = [1]*452 + [0]*(1500-452)
s3ex11_alpha = 0.05
s3ex11_p = compare_mean(s3ex11_l1, s3ex11_l2, -1, -1, s3ex11_alpha, TestType.DOUBLE_SIDED)
print("H0 – średnie równe")
print("Odrzucamy H0" if s3ex11_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 12")
s3ex12_l1 = [0.22, 0.18, 0.16, 0.19, 0.20, 0.23, 0.17, 0.25]
s3ex12_l2 = [0.28, 0.25, 0.20, 0.30, 0.19, 0.26, 0.28, 0.24]
print("H0 – próbka 1 >= próbka 2 ")
s3ex12_alpha = 0.05

s3ex12_p = compare_mean(s3ex12_l1, s3ex12_l2, -1, -1, s3ex12_alpha, TestType.LEFT_SIDED)
print("Odrzucamy H0" if s3ex12_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 13")
s3ex13_l = [20, 16, 14, 14, 10, 16]
s3ex13_alpha = 0.05
s3ex13_p = check_uniform(s3ex13_l, s3ex13_alpha)
print("H0 – prawdopodobieństwa są równe")
print("Odrzucamy H0" if s3ex13_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 14")
s3ex14_l = [50, 100, 80, 40, 20, 10]
s3ex14_lambda = mean([0]*50+[1]*100+[2]*80+[3]*40+[5]*20+[6]*10)  # ?
s3ex14_alpha = 0.05
s3ex14_p = check_pois(s3ex14_l, s3ex14_lambda, s3ex14_alpha)
print("H0 – rozkład z Poissona")
print("Odrzucamy H0" if s3ex14_p else "Nie ma podstaw by odrzucić H0")
print()


print("Zadanie 15")
s3ex15_l = [10, 27, 29, 16, 8, 7]
s3ex15_lambda = mean([0]*10+[1]*27+[2]*29+[3]*16+[5]*8+[6]*7)  # ?
s3ex15_alpha = 0.05
s3ex15_p = check_pois(s3ex15_l, s3ex15_lambda, s3ex15_alpha)
print("H0 – rozkład z Poissona")
print("Odrzucamy H0" if s3ex15_p else "Nie ma podstaw by odrzucić H0")

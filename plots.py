from typing import Generator
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import random
from scipy.optimize import curve_fit

import math
import zaremba
from zaremba import \
    count_zaremba_for, all_true, count_zaremba_up_to, NumFilter, is_prime, \
    all_lengths, find_zaremba_up_to, timed_function

def objective(x, a, b, c):
    return a / np.log(x) + b

@timed_function
def plot_zaremba_count_up_to(max_denominator: int, bound: int, num_filter: NumFilter = all_true):
    """
    For all denominators up to max_denominator, compute how many Zaremba rationals each one has, satisfying the given
    filter (e.g. all numerators, prime numerators, etc.).
    """
    counters = np.array(count_zaremba_up_to(max_denominator, bound, num_filter))

    denominators = range(1, max_denominator)
    xx = denominators[1:]
    yy = np.log(counters[1:])/np.log(xx)

    # for c in range(1, 200):
    #     print(f'{xx[c]}: counter={counters[c+1]}, n({c})={yy[c]}')

    count_value = 5
    # xx = [x for x,b in zip(xx, counters[1:] == 5) if b]
    # yy = [y for y,b in zip(yy, counters[1:] == 5) if b]
    # popt, _ = curve_fit(objective, xx, yy)
    # print(popt)
    # y_line = objective(xx, *popt)

    # plt.scatter(np.log(xx), yy, label=f'zaremba count({bound})', color='blue', s=1)
    # plt.plot(xx, y_line, '--', color='red')
    plt.figure()
    plt.xlabel ('denominators')
    plt.ylabel (f'log(#{bound}-Zaremba)/log(denominator)')
    plt.scatter(xx, yy, label=f'zaremba count({bound})', color='green', s=0.1)
    plt.figure()
    plt.xlabel ('denominators')
    plt.ylabel (f'# {bound}-Zaremba numerators')
    plt.scatter(denominators, counters, label=f'zaremba count({bound})', color='green', s=0.1)
    plt.show()


# # Example:
plot_zaremba_count_up_to(50000, 5)#0.775
# plot_zaremba_count_up_to(10000, 6, zaremba.is_prime)
exit(0)

def plot_zaremba_count(
        denominators: Generator[int, None, None], bound: int, num_filter: NumFilter = all_true):
    """
    For a choice of denominators, compute how many Zaremba rationals each one has, satisfying the given
    filter (e.g. all numerators, prime numerators, etc.).
    """
    counters = count_zaremba_for(denominators, bound, num_filter)
    xx = denominators
    yy = np.log(counters)/np.log(xx)

    plt.scatter(xx, yy, label=f'zaremba count({bound})', color='blue', s=1)
    plt.xlabel ('denominators')
    plt.ylabel (f'log(#{bound}-Zaremba)/log(denominator)')

    # plt.scatter(denominators, np.power(counters, 1.5), label=f'zaremba count({bound})', color='blue')
    plt.show()


# # Example:
# # 100 random numbers in [1,100000]. Note that right now primes are computed only up to 100000
# import random
# denominators = [random.randint(1,100000) for _ in range(100)]
# plot_zaremba_count(denominators, 6, zaremba.is_prime)


def plot_zaremba_count_with_prime_denominators(max_denominator: int, bound: int, num_filter: NumFilter = all_true):
    counters = count_zaremba_up_to(max_denominator, bound, num_filter)
    denominators = range(1, max_denominator)
    plt.title(f'count Zaremba({bound}) - prime denominator')
    plt.scatter(denominators, np.power(counters, 1.5), label=f'zaremba count({bound})', color='blue')

    denominators = [p for p in range(1, max_denominator) if is_prime(p)]
    counters = count_zaremba_for(denominators, bound, num_filter)
    plt.scatter(denominators, np.power(counters, 1.5), label=f'zaremba count({bound})', color='red')
    plt.show()


# # Example
# plot_zaremba_count_with_prime_denominators(10000, 6)
# plot_zaremba_count_with_prime_denominators(10000, 6, zaremba.is_prime)

def plot_zaremba_rationals_up_to(max_denominator: int, bound: int):
    """
    for all (i,j) coprime with 0 <= j < i <= max_denominator, plot a point if j/i is Zaremba(bound).
    Blue point if j is prime and green otherwise
    """
    zaremba_rationals = find_zaremba_up_to(max_denominator, bound)
    all_points = sum([
        [(i, j) for j in range(1, i) if zaremba_rationals[i][j]]
        for i in range(1, max_denominator)], [])
    prime_points = [(i, j) for i, j in all_points if zaremba.is_prime(j)]

    plt.scatter([x for x, _ in all_points], [y/x for x, y in all_points], color='green', marker='.')
    plt.scatter([x for x, _ in prime_points], [y/x for x, y in prime_points], color='blue', marker='.')
    plt.xlabel('denominators')
    plt.ylabel(f'{bound}-Zaremba numerators')
    plt.show()


# plot_zaremba_rationals_up_to(300, 6)


def plot_length_distribution(denominator: int, bins: int = 25, num_filter: NumFilter = all_true):
    x = np.array(all_lengths(denominator=denominator, num_filter=num_filter), dtype=float)
    mean = np.mean(x)
    x -= np.mean(x)
    std = np.std(x)
    x /= std
    mean_ratio = '%.3f' % (mean * math.pi ** 2 / (12 * np.log(2) * np.log(denominator)))
    std = '%.3f' % std
    print(f'mean/ln(denominator)={mean_ratio}, {std=}')
    plt.figure(f'distribution_primes_q={denominator}')
    plt.hist(x=x, bins=bins, density=True, color='green')
    plt.title(f'P: q={denominator}, mean_ratio={mean_ratio}, std={std}')

    mu = 0
    variance = 1
    sigma = math.sqrt(variance)
    x = np.linspace(mu - 3 * sigma, mu + 3 * sigma, 100)
    plt.plot(x, stats.norm.pdf(x, mu, sigma))

    plt.show()

# # Example
# plot_length_distribution(100000),
# plot_length_distribution(100000, num_filter=zaremba.is_prime)

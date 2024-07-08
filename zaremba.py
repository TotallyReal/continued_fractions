from time import time
from typing import Set, List, Generator, Callable, Dict
import numpy as np
from functools import wraps
import cProfile
import primes

# --------------------------- common function ---------------------------

NumFilter = Callable[[int], bool]

# Use to see which function runs the fastest
def timed_function(func):
    @wraps(func)
    def wrapper(*arg, **kwarg):
        t = time()
        result = func(*arg, **kwarg)
        print(f'it took {time() - t} seconds to run {func.__name__}')
        return result

    return wrapper


def list_and(aaa, bbb):
    return [a and b for a, b in zip(aaa, bbb)]


def compare(s1: Set, s2: Set) -> bool:
    """
    Compare two sets
    """
    s2 = s2.copy()
    for x in s1:
        if x not in s2:
            print(f'{x} is in the first set, but not in the second')
            return False
        s2.remove(x)
    if len(s2) > 0:
        print(f'{s2.pop()} is in the second set but not in the first')
        return False
    print('The two sets are the same')
    return True


def all_true(p: int) -> bool:
    return True


def is_prime(p: int) -> bool:
    return primes.is_prime[p]


# --------------------------- continued fraction coefficients ---------------------------


def cf(numerator: int, denominator: int) -> Generator[int, None, None]:
    """
    A generator which returns the coefficients in the continued fraction numerator/denominator.
    If numerator, denominator are not coprime, also return at the end -gcd(numerator, denominator).

    example:
    cf(5,9)    -> 1, 1, 4
    cf(10,18)  -> 1, 1, 4, -2
    """
    numerator = abs(numerator)
    denominator = abs(denominator)
    while numerator > 0:
        q, remainder = divmod(denominator, numerator)
        yield q
        numerator, denominator = remainder, numerator
    if denominator == 1:
        return
    yield -denominator


# --------------------------- single continued fraction bounded ---------------------------

def cf_is_bounded(numerator: int, denominator: int, bound: int) -> bool:
    """
    Returns True if numerator and denominator are coprime and the coefficients in the continued fraction
    of numerator/denominator are bounded <= by the given bound. Otherwise returns false.
    """
    c = -1
    for c in cf(numerator, denominator):
        if c > bound:
            return False
    return c > 0  # if c<=0, then (numerator, denominator) > 1 are not coprime


"""
--------------------------- Zaremba count ---------------------------

The following are all versions of counting rational which have bounded continued fractions (for a given bound), 
which I call 'Zaremba count' after Zaremba's conjecture.

In each of them we have:
    1) The bound 
    2) The set of denominators m, for which we consider rational n/m with 0<=n<m (namely in [0,1) ).
    3) An extra filter on the numbers (e.g. n is prime).
"""


@timed_function
def count_zaremba_for(
        denominators: Generator[int, None, None], bound: int, num_filter: NumFilter = all_true):
    """
    Counts Zaremba rational, with the simple bound check function cf_is_bounded.
    """
    return [
        sum([cf_is_bounded(numerator=numerator, denominator=denominator, bound=bound)
             for numerator in range(0, denominator) if num_filter(numerator)])
        for denominator in denominators
    ]


"""
--------------------------- range Zaremba count ---------------------------

Counting Zaremba rationals for denominators in a range [1, max_denominator]
By saving all the Zaremba rational, while it costs more in space, it can run faster.
Tried all sort of functions, to find one which is fast. Can probably get even better than the ones here.

The main idea here is that for a rational number j/i with 0<j<i, we can write i = q*j + r for division 
with remainder. We then have that 

j / i = j / (q * j + r) = 1 / (q + (r/j))

This means that j/i has continued fraction bounded by C iff q<=C and r/j has continued fraction bounded
by C. This suggests using the Zaremba count for simpler rationals (smaller numerator\denominator) to 
compute the Zaremba count for more complex rationals. 
"""

@timed_function
def find_zaremba_up_to1(max_denominator: int, bound: int) -> Set:
    # Simple counting. Returns set of tuples (j, i) for j/i Zaremba(bound).
    bounded_rcf = {(0, 1)}
    for i in range(2, max_denominator):
        for j in range(1, i):
            q, remainder = divmod(i, j)
            if q <= bound and (remainder, j) in bounded_rcf:
                bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to2(max_denominator: int, bound: int) -> Set:
    # just skip the small numerators, to avoid the if statement on the numerator.
    # Returns set of tuples (j, i) for j/i Zaremba(bound).
    bounded_rcf = {(0, 1)}
    for i in range(2, max_denominator):
        for j in range(1 + int(i // (bound + 1)), i):
            remainder = i % j
            if (remainder, j) in bounded_rcf:
                bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to3(max_denominator: int, bound: int) -> Set:
    # try to use numpy
    # Returns set of tuples (j, i) for j/i Zaremba(bound).
    bounded_rcf = {(0, 1)}
    all_denominators = np.arange(1, max_denominator)
    for i in range(2, max_denominator):
        numerators = np.full(i - 1, i, dtype=int)
        denominators = all_denominators[:i - 1]
        quotients, remainders = np.divmod(numerators, denominators)
        for q, r, j in zip(quotients, remainders, denominators):
            if q <= bound and (r, j) in bounded_rcf:
                bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to4(max_denominator: int, bound: int) -> Set:
    # flip the counting order - first go over numerators and then denominators
    bounded_rcf = {(0, 1)}
    for j in range(1, max_denominator):
        for i in range(j + 1, min(max_denominator, j * (bound + 1))):
            if (i % j, j) in bounded_rcf:
                bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to5(max_denominator: int, bound: int) -> Set:
    # flip the counting order - first go over numerators and then denominators
    # remainder only goes from 1 to j, avoid "general" modulo % operation
    bounded_rcf = {(0, 1)}
    for j in range(1, max_denominator):
        remainder = 0
        for i in range(j + 1, min(max_denominator, j * (bound + 1))):
            remainder = (remainder + 1) % j
            if (remainder, j) in bounded_rcf:
                bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to6(max_denominator: int, bound: int) -> Set:
    # flip the counting order - first go over numerators and then denominators. Also, try to avoid using
    # the modulo % operation
    bounded_rcf = {(1, k) for k in range(2, bound + 1)}
    bounded_rcf.add((0, 1))

    for j in range(2, max_denominator):
        upper_bound = min(max_denominator, j * (bound + 1))
        for remainder in range(1, j):
            if (remainder, j) in bounded_rcf:
                for i in range(j + remainder, upper_bound, j):
                    bounded_rcf.add((j, i))
    return bounded_rcf


@timed_function
def find_zaremba_up_to7(max_denominator: int, bound: int) -> Dict[int, Set[int]]:
    # one line set creation, seems to be working faster.
    # Returns dictionary where denominator->{numerators} so that numerator/denominator are Zaremba(bound)
    bounded_rcf = {1: {0}}
    for i in range(2, max_denominator):
        bounded_rcf[i] = {j for j in range(1 + int(i // (bound + 1)), i) if i % j in bounded_rcf[j]}
    return bounded_rcf


def dict_int_int_to_standard(d: Dict[int, Set[int]]) -> Set:
    all_sets = [{(j, i) for j in d[i]} for i in d]
    s = set()
    s.update(*all_sets)
    return s


@timed_function
def find_zaremba_up_to8(max_denominator: int, bound: int) -> Dict[int, List[bool]]:
    # use list generation instead of dictionaries. Sometimes faster
    # Returns dictionary d where denominator->[num is zaremba] so that d[i][j] == True  <=>  j/i is Zaremba(bound)
    bounded_rcf = {1: [True]}
    for i in range(2, max_denominator):
        initial = 1 + i // (bound + 1)
        bounded_rcf[i] = [False] * initial + [bounded_rcf[j][i % j] for j in range(initial, i)]
    return bounded_rcf


def dict_int_bool_to_standard(d: Dict[int, List[int]]) -> Set:
    all_sets = [{(j, i) for j in range(0, i) if d[i][j]} for i in d]
    s = set()
    s.update(*all_sets)
    return s


# Compare the different counting techniques to validate them (at least for simpler small rationals).
def compare_zaremba_counters(max_denominator: int, bound: int):
    s1 = find_zaremba_up_to1(max_denominator=max_denominator, bound=bound)
    s2 = find_zaremba_up_to2(max_denominator=max_denominator, bound=bound)
    s3 = find_zaremba_up_to3(max_denominator=max_denominator, bound=bound)
    s4 = find_zaremba_up_to4(max_denominator=max_denominator, bound=bound)
    s5 = find_zaremba_up_to5(max_denominator=max_denominator, bound=bound)
    s6 = find_zaremba_up_to6(max_denominator=max_denominator, bound=bound)
    d7 = find_zaremba_up_to7(max_denominator=max_denominator, bound=bound)
    s7 = dict_int_int_to_standard(d7)
    d8 = find_zaremba_up_to8(max_denominator=max_denominator, bound=bound)
    s8 = dict_int_bool_to_standard(d8)

    compare(s1, s2)
    compare(s1, s3)
    compare(s1, s4)
    compare(s1, s5)
    compare(s1, s6)
    compare(s1, s7)
    compare(s1, s8)


# compare_zaremba_counters(max_denominator=5000, bound=6)

# usually the 7 and 8 versions of find_zaremba_up_to are the fastest, with 8 usually the best
find_zaremba_up_to = find_zaremba_up_to8

@timed_function
def count_zaremba_up_to(max_denominator: int, bound: int, num_filter: NumFilter = all_true):
    zaremba_rationals = find_zaremba_up_to(max_denominator=max_denominator, bound=bound)
    return [
        sum([num_filter(numerator) and zaremba_rationals[denominator][numerator] for numerator in range(denominator)])
        for denominator in range(1, max_denominator)]


# --------------------------- continued fraction length ---------------------------


def cf_length(numerator: int, denominator: int) -> int:
    numerator = abs(numerator)
    denominator = abs(denominator)
    counter = 0
    while numerator > 0:
        numerator, denominator = denominator % numerator, numerator
        counter += 1
    if denominator == 1:
        return counter
    return -1


def cf_length2(numerator: int, denominator: int) -> int:
    counter = 0
    c = 0
    for c in cf(numerator, denominator):
        counter += 1
    if c < 0:  # gcd(numerator, denominator)>1
        return -1
    return counter


@timed_function
def all_lengths(denominator: int, num_filter: NumFilter = all_true) -> List[int]:
    """
    Given the denominator m, run over all n/m in [0,1) satisfying the filter, and returns
    a list of their continued fractions lengths.
    """
    lengths = [cf_length(numerator, denominator) for numerator in range(1, denominator) if num_filter(numerator)]
    return [length for length in lengths if length != -1]

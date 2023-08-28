from time import time
import sympy

prime_size = 100000  # compute all the primes until this number

# A list of all primes (and 1)
print(f'Computing primes up to {prime_size}')
t = time()
primes = [1]+list(sympy.primerange(0, prime_size+1))

# call is_prime[p] to find out if p is prime. Only works up to prime_size, and counts 1 as prime.
is_prime = [False] * (prime_size + 1)
for p in primes:
    is_prime[p] = True
print(f'Finished computing primes (in {time()-t} seconds)')

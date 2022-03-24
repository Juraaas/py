from typing import Any, List, Tuple
from math import sqrt

def numbers(n: int) -> int:
    if n < 0:
        return
    print(n)
    numbers(n - 1)


def fib(n: int) -> int:
    if n == 0:
        return 0
    if n == 1:
        return 1
    return fib(n - 1) + fib(n - 2)


def power(number: int, n: int) -> int:
    if n == 0:
        return 1
    if n == 1:
        return number
    return number*power(number, n-1)


def reverse(s: str) -> str:
    if len(s) == 0:
        return s
    else:
        return reverse(s[1:]) + s[0]


def factorial(n: int) -> int:
    if n == 0:
        return 1
    if n == 1:
        return 1
    return n*factorial(n-1)


def prime(n: int, i: int = 2) -> bool:

    if n <= 1:
        return False
    if i * i > n:
        return True
    if n % i == 0:
        return False
    return prime(n, i+1)


def list_sum(lista: List):
    if len(lista) == 1:
        return lista[0]
    else:
        return lista[0] + list_sum(lista[1:])


def list_sum2(lista: List):
    total = 0
    for ele in lista:
        if type(ele) == type([]):
            total = total + list_sum2(ele)
        else:
            total = total + ele
    return total


def num_sum(n: int):
    if n == 0:
        return 0
    else:
        return n % 10 + num_sum(int(n/10))


def sum_series(n: int):
    if n < 1:
        return 0
    else:
        return n + sum_series(n - 2)


def gcd(a: int, b: int):
    low = min(a, b)
    high = max(a, b)
    if low == 0:
        return high
    elif low == 1:
        return 1
    else:
        return gcd(low, high % low)


def pascal(n: int):
    if n == 1:
        return [1]
    else:
        line = [1]
        previous_line = pascal(n-1)
        for i in range(len(previous_line)-1):
            line.append(previous_line[i] + previous_line[i+1])
        line += [1]
    return line


def fib_pascal(n: int, fib_pos: int):
    if n == 1:
        line = [1]
        fib_sum = 1 if fib_pos == 0 else 0
    else:
        line = [1]
        (previous_line, fib_sum) = fib_pascal(n-1, fib_pos+1)
        for i in range(len(previous_line)-1):
            line.append(previous_line[i] + previous_line[i+1])
        line += [1]
        if fib_pos < len(line):
            fib_sum += line[fib_pos]
    return line, fib_sum


def fib2(n: int):
    return fib_pascal(n, 0)[1]


for i in range(1,10):
    print(fib2(i))


def sito(n: int):
    primes = list(range(2, n+1))
    maxi = sqrt(n)
    num = 2
    while num < maxi:
        i = num
        while i <= n:
            i += num
            if i in primes:
                primes.remove(i)
        for j in primes:
            if j > num:
                num = j
                break
    return primes


print(sito(100))





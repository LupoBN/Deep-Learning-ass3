import random

import math

from gen_examples import *


def is_even(n):
    if n % 2 == 0:
        return True
    else:
        return False


def is_divided_by_three(n):
    if n % 3 == 0:
        return True
    else:
        return False


def does_contain_three(n):
    n_str = str(n)
    for char in n_str:
        if char == "3":
            return True
    return False


def is_prime(n):
    '''check if integer n is a prime'''
    # make sure n is a positive integer
    n = abs(int(n))
    # 0 and 1 are not primes
    if n < 2:
        return False
    # 2 is the only even prime number
    if n == 2:
        return True
    # all other even numbers are not primes
    if not n & 1:
        return False
    # range starts with 3 and only needs to go up the squareroot of n
    # for all odd numbers
    for x in range(3, int(n ** 0.5) + 1, 2):
        if n % x == 0:
            return False
    return True


def is_power_of_two(n):
    if n != 0 and ((n & (n - 1)) == 0):
        return True
    else:
        return False


def in_or_out(in_lanauage, start=1, end=300, same_ratio=True):
    in_lang, out = list(), list()
    for j in range(start, end):
        if in_lanauage(j):
            in_lang.append(j)
        else:
            out.append(j)
    if same_ratio:
        out = random.sample(out, len(in_lang))
    return in_lang, out


def in_or_out_couple(start=1, end=300):
    in_lang, out = list(), list()
    for i in range(start, end):
        in_lang.append((i, int(math.pow(i, 2))))
        random_a = random.randint(start, end)
        random_b = random.randint(start, end)
        out.append((random_a, random_b))
    return in_lang, out


def create_fibonachi_sequence(first, second, end=300):
    sequence = str(first) + "," + str(second)
    for i in range(end):
        sequence += ","
        third = first + second
        next_num = str(third)
        sequence += next_num
        first = second
        second = third
    return sequence


def create_non_fibonachi_random_sequence(first, second, end=300, maximum_fibonachi_num=200000000):
    sequence = str(first) + "," + str(second)
    for i in range(end):
        sequence += ","
        third = random.randint(0, maximum_fibonachi_num)
        next_num = str(third)
        sequence += next_num
    return sequence


def create_non_fibonachi_sequence(first, second, end=300, add_extra=1):
    sequence = str(first) + "," + str(second)
    for i in range(end):
        sequence += ","
        third = first + second + add_extra
        next_num = str(third)
        sequence += next_num
        first = second
        second = third
    return sequence


def create_binary_language(in_language, start=1, end=1500):
    in_lang, out = in_or_out(in_language, start, end)
    data = [(bin(num)[2:], 1) for num in in_lang]
    data += [(bin(num)[2:], 0) for num in out]
    return data


def create_fibonaci_language(num_of_examples, average_sequence_length, maximum_fibonachi_num=100):
    data = list()
    maxim = maximum_fibonachi_num * maximum_fibonachi_num * 3

    for i in range(num_of_examples):
        rand_first = np.random.randint(maximum_fibonachi_num)
        rand_second = random.randint(rand_first, maximum_fibonachi_num)
        rand_end = random.randint(1, average_sequence_length)
        fibo = create_fibonachi_sequence(rand_first, rand_second, rand_end)
        data.append((fibo, 1))
        if i % 5 != 0 and i % 5 != 1:
            data.append((create_non_fibonachi_sequence(rand_first, rand_second, rand_end, add_extra=i % 20), 0))
        elif i % 5 == 1:
            numbers = fibo.split(',')
            first = int(numbers[-1])
            second = int(numbers[-2])
            third = first + second
            fourth = second + third
            non_fibo = fibo + "," + create_non_fibonachi_random_sequence(third, fourth, rand_end,
                                                                   maximum_fibonachi_num=maxim)
            data.append((non_fibo, 0))
        else:
            data.append((create_non_fibonachi_random_sequence(rand_first, rand_second, rand_end,
                                                              maximum_fibonachi_num=maxim), 0))

    return data


def create_a_language(in_language, start=1, end=500):
    in_lang, out = in_or_out(in_language, start, end)
    data = generate_examples_by_rules(["a"], in_lang, 1)
    data += generate_examples_by_rules(["a"], out, 0)
    return data


def create_ab_language(start=1, end=500, letters=["a", "b"]):
    in_lang, out = in_or_out_couple(start, end)
    data = list()
    for i in range(len(in_lang)):
        data.append((generate_limited_sentence(letters, in_lang[i]), 1))
        data.append((generate_limited_sentence(letters, out[i]), 0))

    return data


if __name__ == '__main__':
    data = create_fibonaci_language(500, 10)
    random.shuffle(data)
    split = int(0.7 * len(data))

    train, dev = data[:split], data[split:]
    write_file("train", train, write_examples_parser, "\n")
    write_file("dev", dev, write_examples_parser, "\n")

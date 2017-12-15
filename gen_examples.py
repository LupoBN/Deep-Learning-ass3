import numpy as np
from Utils import *


def generate_letter_sequence(letter):
    sequence = letter
    rand_num = 0
    while rand_num != 1:
        rand_num = np.random.randint(10)
        sequence += letter
    return sequence


def generate_number_sequence():
    sequence = str()
    rand_num = 0
    while rand_num != 1:
        rand_num = np.random.randint(10)

        digit_sequeunce_num = str(np.random.randint(9) + 1)
        sequence += digit_sequeunce_num
    return sequence


def generate_examples_by_rules(letters, rules, label):
    data = list()
    for rule in rules:
        data.append((generate_limited_sentence(letters, [rule]), label))
    return data


def generate_example(letters, numbers=True):
    sequence = str()
    for letter in letters:
        if numbers:
            sequence += generate_number_sequence()
        sequence += generate_letter_sequence(str(letter))
    return sequence


def generate_sentences(number_of_examples, letters, label, numbers=True):
    data = list()
    for i in range(0, number_of_examples):
        data.append((generate_example(letters, numbers), label))
    return data


def generate_limited_sentence(letters, number_of_appearnces):
    sentence = str()
    for i, letter in enumerate(letters):
        sentence += "".join([letter] * number_of_appearnces[i])
    return sentence


def even_uneven(start=1, end=300):
    even, uneven = list(), list()
    for j in range(start, end):
        if j % 2 == 0:
            even.append(j)
        else:
            uneven.append(j)
    return even, uneven


def create_the_even_language():
    even, uneven = even_uneven(1, 750)
    data = generate_examples_by_rules(["a"], even, 1)
    data += generate_examples_by_rules(["a"], uneven, 0)
    np.random.shuffle(data)
    split = int(0.7 * len(data))
    train, dev = data[:split], data[split:]
    write_file("train", train, write_examples_parser, "\n")
    write_file("dev", dev, write_examples_parser, "\n")


if __name__ == '__main__':
    good_example_letters = ["a", "b", "c", "d"]
    bad_example_letters = ["a", "c", "b", "d"]
    train = generate_sentences(5000, good_example_letters, 1)
    train += generate_sentences(5000, bad_example_letters, 0)
    dev = generate_sentences(500, good_example_letters, 1)
    dev += generate_sentences(500, bad_example_letters, 0)
    write_file("train", train, write_examples_parser, "\n")
    write_file("dev", dev, write_examples_parser, "\n")

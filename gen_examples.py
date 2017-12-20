import numpy as np
from Utils import *

def generate_limited_sentence(letters, number_of_appearnces):
    sentence = str()
    for i, letter in enumerate(letters):
        sentence += "".join([letter] * number_of_appearnces[i])
    return sentence

def generate_examples_by_rules(letters, rules, label):
    data = list()
    for rule in rules:
        data.append((generate_limited_sentence(letters, [rule]), label))
    return data

def generate_letter_sequence(letter):
    sequence = letter
    rand_num = 0
    while rand_num != 1:
        rand_num = np.random.randint(10)
        sequence += letter
    return sequence


def generate_number_sequence(start_from = 1):
    sequence = str()
    rand_num = 0
    while rand_num != 1:
        rand_num = np.random.randint(10)

        digit_sequeunce_num = str(np.random.randint(9) + start_from)
        sequence += digit_sequeunce_num
    return sequence


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


if __name__ == '__main__':
    good_example_letters = ["a", "b", "c", "d"]
    bad_example_letters = ["a", "c", "b", "d"]
    train = generate_sentences(500, good_example_letters, 1)
    train += generate_sentences(500, bad_example_letters, 0)
    dev = generate_sentences(50, good_example_letters, 1)
    dev += generate_sentences(50, bad_example_letters, 0)
    write_file("train", train, write_examples_parser, "\n")
    write_file("dev", dev, write_examples_parser, "\n")

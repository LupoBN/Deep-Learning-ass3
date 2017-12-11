import numpy as np


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


def generate_example(letters):
    sequence = str()
    for letter in letters:
        sequence += generate_number_sequence()
        sequence += generate_letter_sequence(letter)
    return sequence


if __name__ == '__main__':
    good_example_letters = ["a", "b", "c", "d"]
    bad_example_letters = ["a", "c", "b", "d"]
    generate_example(good_example_letters)
    generate_example(bad_example_letters)

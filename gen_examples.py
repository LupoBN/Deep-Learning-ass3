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

if __name__ == '__main__':
    good_example_letters = ["a", "b", "c", "d"]
    bad_example_letters = ["a", "c", "b", "d"]
    generate_example(good_example_letters)
    generate_example(bad_example_letters)
import copy
from collections import Counter


def parse_tag_reading(lines, seperator, lower=False):
    words = list()
    labels = list()
    sentence = list()
    sentence_labels = list()
    for line in lines:
        if line != '':
            words_labels = line.split()
            if lower:
                words_labels[0] = words_labels[0].lower()
            sentence.append(words_labels[0])
            if len(words_labels) > 1:
                sentence_labels.append(words_labels[1])
        else:
            sentence = ["^^^^^"] + sentence + ["$$$$$"]
            sentence_labels = ["Start-"] + sentence_labels + ["End-"]
            words.append(copy.deepcopy(sentence))
            labels.append(copy.deepcopy(sentence_labels))
            sentence = list()
            sentence_labels = list()
    return words, labels


def parse_vocab_words_reading(lines, seperator=None, lower=False):
    words = [line for line in lines]
    W2I = {key: value for value, key in enumerate(words)}

    return words, W2I


def parse_vocab_reading(lines, seperator=None, lower=False):
    words = [line for line in lines]
    W2I = {key: value for value, key in enumerate(words)}
    return W2I


def get_power_two_mapping(max_value):
    W2I = {str(i) + "a": i for i in range(0, max_value)}
    return W2I


def sub_words_mapping(sentences, start, most_to_take=5002):
    prefixes_words = [["Pre-" + word[0:3] if len(word) >= 3 else "lessthanthree"] for sentence in sentences for word in
                      sentence[0]]
    suffixes_words = [["Suf-" + word[-4:-1] if len(word) >= 3 else "lessthanthree"] for sentence in sentences for word
                      in sentence[0]]
    count_prefixes = count_uniques(prefixes_words)
    count_suffixes = count_uniques(suffixes_words)
    del count_prefixes["lessthanthree"]
    del count_suffixes["lessthanthree"]
    possibles_prefixes = set([x for x, l in count_prefixes.most_common(most_to_take)])
    possibles_suffixes = set([x for x, l in count_suffixes.most_common(most_to_take)])
    possibles_prefixes.add("Pre-UNK")
    possibles_suffixes.add("Suf-UNK")

    sub_map = dict()
    for pre in possibles_prefixes:
        if pre not in sub_map:
            sub_map[pre] = start
            start += 1
    for suf in possibles_suffixes:
        if suf not in sub_map:
            sub_map[suf] = start
            start += 1

    return sub_map


def write_examples_parser(content, seperator):
    output_str = str()
    for sample in content:
        output_str += str(sample[1]) + "\t" + sample[0] + seperator
    return output_str


def read_examples_parser(content, seperator, lower=False):
    examples = list()
    for sample in content:
        new_sample = sample.split(seperator)
        example = (new_sample[1], new_sample[0])
        examples.append(example)
    return examples


def read_file(file_name, parse_func, seperator=None, lower=False):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    file.close()
    return parse_func(lines, seperator, lower)


def write_file(file_name, content, parse_func, seperator):
    file = open(file_name, 'w')
    file.write(parse_func(content, seperator))
    file.close()


def create_mapping(data, ignore_elements=None, most_to_take=15000):
    count = count_uniques(data)
    possibles = set([x for x, l in count.most_common(most_to_take)])
    if ignore_elements != None:
        possibles = possibles.difference(ignore_elements)
    else:
        possibles.add("UUUNKKK")

    return {f: i for i, f in enumerate(list(sorted(possibles)))}


def count_uniques(sentences):
    fc = Counter()
    for words in sentences:
        fc.update(words)
    return fc

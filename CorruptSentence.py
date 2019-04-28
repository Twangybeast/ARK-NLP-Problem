import random
import time
import unicodedata

import ErrorClassifier

PATH_IN = 'train.txt'
PATH_OUT = 'part2.txt'


def corrupt_arrange(original):
    # pick 2 random tokens and scramble them
    tokens = original.split()
    if len(tokens) <= 1:
        return original
    for i in range(1000):
        i1 = random.randint(0, len(tokens) - 1)
        i2 = random.randint(0, len(tokens) - 1)
        # Make sure they aren't the same
        if tokens[i1] != tokens[i2]:
            break
    temp_token = tokens[i1]
    tokens[i1] = tokens[i2]
    tokens[i2] = temp_token
    return ' '.join(tokens)


def corrupt_add(original):
    tokens = original.split()
    word = random.choice(vocab)
    i = random.randint(0, len(tokens))
    tokens = tokens[:i] + [word] + tokens[i:]
    return ' '.join(tokens)


def corrupt_remove(original):
    tokens = original.split()
    i = random.randint(0, len(tokens) - 1)
    tokens = tokens[:i] + tokens[i + 1:]
    return ' '.join(tokens)


def corrupt_typo(original):
    tokens = original.split()
    # switch some letters up in that word
    if len(original) == 1:
        return original
    for i in range(1000):
        i = random.randint(0, len(tokens) - 1)
        word = list(tokens[i])
        c1 = random.randint(0, len(word) - 1)
        c2 = random.randint(0, len(word) - 1)
        if word[c1] != word[c2]:
            break
    temp = word[c1]
    word[c1] = word[c2]
    word[c2] = temp
    tokens[i] = ''.join(word)
    return ' '.join(tokens)


def corrupt_replace(original):
    tokens = original.split()
    for i in range(1000):
        word = random.choice(vocab)
        i = random.randint(0, len(tokens) - 1)
        if tokens[i] != word:
            break
    tokens = tokens[:i] + [word] + tokens[i+1:]
    return ' '.join(tokens)


def choose_errors(original):
    while True:
        # Choose 2 random error types to use
        error_choices = random.sample(ErrorClassifier.ERROR_TYPES, k=2)

        # prevent add & remove from being the decided errors, they might cancel out
        if not ('ADD' in error_choices and 'REMOVE' in error_choices):
            break
    if len(original.split()) <= 1:
        return ['REPLACE', 'ADD']
    return error_choices


def corrupt_text(original):
    corrupted = original

    error_choices = choose_errors(original)
    for error_type in error_choices:
        corrupted = globals()['corrupt_%s' % error_type.lower()](corrupted)
    return corrupted

def main():
    random.seed(123) # Consistency across runtimes
    with open(PATH_IN, encoding='utf-8') as file_in, open(PATH_OUT, 'w', encoding='utf-8') as file_out:
        progress = 0
        start_time = time.time()
        lines_processed = 0
        for line in file_in:
            progress += 1

            line = line.strip()
            line = unicodedata.normalize('NFKD', line)
            p1, p2 = line.split('\t')


            while True:
                corrupted = corrupt_text(p1)
                if p1.strip() == corrupted.strip():
                    continue
                if p2.strip() == corrupted.strip():
                    continue
                break

            file_out.write("{}\t{}\n".format(p1, corrupted))

            assert p1.strip() != corrupted.strip()
            assert p2.strip() != corrupted.strip()

            # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
            # console
            lines_processed += 1
            if progress % 100 == 0:
                print('\rProgress: [{}] Lines per second: [{}]'
                      .format(lines_processed, (lines_processed / (time.time() - start_time)))
                      , end='')


if __name__ == '__main__':
    vocab = tuple(ErrorClassifier.load_words_list())
    main()

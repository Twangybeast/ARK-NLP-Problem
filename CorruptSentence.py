import random
import time
import unicodedata

import ErrorClassifier

PATH_IN = 'train.txt'
PATH_OUT = 'part2.txt'


def corrupt_arrange(original):
    # pick 2 random tokens and scramble them
    tokens = original.split()
    while True:
        i1 = random.randint(0, len(tokens) - 1)
        i2 = random.randint(0, len(tokens) - 1)
        # Make sure they aren't the same
        if tokens[i1] != tokens[i2]:
            break
    # swap them
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
    i = random.randint(0, len(tokens) - 1)
    # switch some letters up in that word
    word = tokens[i]
    while True:
        c1 = random.randint(0, len(word) - 1)
        c2 = random.randint(0, len(word) - 1)
        if word[c1] != word[c2]:
            break
    temp = word[c1]
    word[c1] = word[c2]
    word[c2] = temp
    tokens[i] = word
    return ' '.join(tokens)


def corrupt_replace(original):
    tokens = original.split()
    word = random.choice(vocab)
    while True:
        i = random.randint(0, len(tokens - 1))
        if tokens[i] != word:
            break
    tokens = tokens[:i] + [word] + tokens[i+1:]
    return ' '.join(tokens)


def choose_errors():
    while True:
        # Choose 2 random error types to use
        error_choices = random.choice(ErrorClassifier.ERROR_TYPES, k=2)

        # prevent add & remove from being the decided errors, they might cancel out
        if not ('ADD' in error_choices and 'REMOVE' in error_choices):
            break
    return error_choices


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

            corrupted = p1

            error_choices = choose_errors()
            for error_type in error_choices:
                corrupted = globals()['corrupt_%s' % error_type.lower()](corrupted)

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

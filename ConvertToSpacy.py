import time
import unicodedata

import spacy
nlp = spacy.load('en_core_web_lg', disable=['parser', 'ner'])


# Converts the entire text file into a spacy file
src_name = 'labeled_test'
parts1 = []
parts2 = []
print('Reading input...')
with open(src_name + '.txt', encoding='utf-8') as file:
    start_time = time.time()
    progress = 0
    for line in file:
        progress += 1

        line = line.strip()
        line = unicodedata.normalize('NFKD', line)
        p1, p2 = line.split('\t')

        parts1.append(p1)
        parts2.append(p2)

        # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
        # console
        if progress % 100 == 0:
            print('\rProgress: [{0}]'.format(progress), end='')
    print()
    print('Finished reading in : [{}]'.format(time.time() - start_time))

doc1 = nlp.pipe(parts1, batch_size=1000)
doc2 = nlp.pipe(parts2, batch_size=1000)

print('Writing output...')
with open(src_name + '.spacy.txt', 'w') as file:
    start_time = time.time()
    progress = 0
    words_count = 0
    assert len(parts1) == len(parts2)
    for d1, d2 in zip(doc1, doc2):
        progress += 1

        words_count += len(d1)
        words_count += len(d2)
        for t in d1:
            file.write(t.tag_ +' ')
        file.write('\t')
        for t in d2:
            file.write(t.tag_ +' ')
        file.write('\n')

        # Display progression in number of samples processed, use random to avoid too many (slow) interactions w/
        # console
        if progress % 10 == 0:
            print('\rProgress: [{}] Words: [{}] WPS: [{}]'.format(progress, words_count, words_count/(time.time() - start_time)), end='')
    print()
    print('Finished saving in : [{}]'.format(time.time() - start_time))
    print('\rWords: [{}] WPS: [{}]'.format(words_count, words_count/(time.time() - start_time)), end='')
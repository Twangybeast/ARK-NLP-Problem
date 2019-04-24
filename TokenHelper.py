import re

import spacy
nlp_tokenizer = spacy.load('en_core_web_lg', disable=['tagger', 'parser', 'ner'])


# Takes two strings and returns two Spacy docs, representing the tokenized strings
def tokenize(p1, p2):
    t1 = nlp_tokenizer(p1)
    t2 = nlp_tokenizer(p2)
    return t1, t2


# Takes a string and returns a list containing only alphanumeric words, using whitespace as a delimiter & not
# including whitespace in the result
def tokenize_pure_words(part):
    part = re.sub('[^a-zA-Z0-9 ]', ' ', part).lower()
    words = part.split()
    return words


# "deltas" are in REPLACE errors. They represent the difference between the part pairs Returns the segmented tokens

# from a REPLACE error (given the tokens of the parts), where each of the original parts is in the format START DELTA
# END, only differing in what the value oF DELTA is
def find_all_delta_from_tokens(t1, t2):
    starting_match = 0
    for i in range(min(len(t1), len(t2))):
        if str(t1[i]) != str(t2[i]):
            starting_match = i
            break

    ending_match = 0
    for i in range(min(len(t1), len(t2))):
        if str(t1[-i - 1]) != str(t2[-i - 1]):
            ending_match = i
            break

    start = t1[:starting_match]
    end = t1[len(t1) - ending_match:]
    return t1[starting_match:len(t1) - ending_match], t2[starting_match:len(t2) - ending_match], start, end


# Like above, but only returns the deltas (not the start or end)
def find_delta_from_tokens(t1, t2):
    delta1, delta2, _, _ = find_all_delta_from_tokens(t1, t2)
    return delta1, delta2

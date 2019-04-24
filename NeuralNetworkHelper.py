
# Some basic constants that may be configured
FILE_NAME = 'train'

PATH_REPLACE_CHECKPOINT = 'checkpoints/%s_replace_w0_d1.ckpt' % FILE_NAME
PATH_ARRANGE_CHECKPOINT = 'checkpoints/%s_arrange_w1_d0.ckpt' % FILE_NAME

# Range of train.txt that is the testing set. Trainer skips these, Tester only works on these line
TESTING_RANGE = (900000, 1000000)


# Loads a file containing all the possible tags in Spacy. Returns a map to convert from a string (Spacy tag) to an
# integer (id)
def load_tags_to_id():
    tags_to_id = {}
    with open('spacy_tags.txt') as tags_list:
        id = 1
        for line in tags_list:
            tag = line.strip()
            tags_to_id[tag] = id
            id += 1
    return tags_to_id


tags_to_id = load_tags_to_id()
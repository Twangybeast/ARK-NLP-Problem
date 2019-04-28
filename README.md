# ARK NLP Problem
The two primary files are Tester.py and Trainer.py, which tests and trains the model respectively. Tester.py, when given the appropriate boolean flag, can output its results to a file. The raw data file used is found in NeuralNetworkHelper.py

Additionally, NNTest.py contains code for training the neural network used for REPLACE errors. (This requires another file to train off of, generated in ExtractReplace.py, which includes only REPLACE sentence pairs)

ConvertToSpacy must be run on any data file used in Tester.py or Trainer.py. This file simply creates an auxiliary data file containing Spacy tags, for the purpose of preprocessing due to the long time it takes.

CorruptSentence, given a file of labeled sentence pairs, outputs a file containing a different corruption, guaranteeing that a) the new sentence pairs are not duplicates of the same sentence and b) they are different from the original file

ErrorClassifier contains code to identify the error of sentence pairs, choosing either ARRANGE, ADD, REMOVE, TYPO, and REPLACE. Note that since ADD/REMOVE are symmetrical, Tester.py uses essentially the same algorithm on both of them. Additionally, ErrorClassifier, when run, will print the frequencies of each ErrorType

NNModels simply contains code defining the architecture of the neural networks.

NeuralNetworkHelper contains some miscellaneous shared variables and function

TokenHelper contains miscellaneous code for processing tokens

## Libraries Used ##
* Spacy
* Tensorflow
* Python Levenshtein

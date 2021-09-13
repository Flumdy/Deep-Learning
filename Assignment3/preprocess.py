import numpy as np
import tensorflow as tf
import numpy as np

##########DO NOT CHANGE#####################
PAD_TOKEN = "*PAD*"
STOP_TOKEN = "*STOP*"
START_TOKEN = "*START*"
UNK_TOKEN = "*UNK*"
FIRST_SPECIAL = "*"
LAST_SPECIAL = "*~"
WINDOW_SIZE = 15
##########DO NOT CHANGE#####################

def pad_english_corpus(english):
	"""
	DO NOT CHANGE:

	argument is a list of ENGLISH sentences. Returns ENGLISH-sents. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end and "*START*" at the beginning.

	:param english: list of English sentences
	:return: list of padded sentences for English
	"""
	ENGLISH_padded_sentences = []
	ENGLISH_sentence_lengths = []
	for line in english:
		padded_ENGLISH = line[:WINDOW_SIZE]
		padded_ENGLISH = [START_TOKEN] + padded_ENGLISH + [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_ENGLISH))
		ENGLISH_padded_sentences.append(padded_ENGLISH)
	return ENGLISH_padded_sentences

def pad_french_corpus(french):
	"""
	DO NOT CHANGE:

	argument is a list of FRENCH sentences. Returns FRENCH-sents. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param french: list of French sentences
	:return: list of padded sentences for French
	"""
	FRENCH_padded_sentences = []
	FRENCH_sentence_lengths = []
	for line in french:
		padded_FRENCH = line[:WINDOW_SIZE]
		padded_FRENCH += [STOP_TOKEN] + [PAD_TOKEN] * (WINDOW_SIZE - len(padded_FRENCH)-1)
		FRENCH_padded_sentences.append(padded_FRENCH)

	return FRENCH_padded_sentences

def pad_corpus(french, english):
	"""
	DO NOT CHANGE:

	arguments are lists of FRENCH, ENGLISH sentences. Returns [FRENCH-sents, ENGLISH-sents]. The
	text is given an initial "*STOP*".  All sentences are padded with "*STOP*" at
	the end.

	:param french: list of French sentences
	:param english: list of English sentences
	:return: A tuple of: (list of padded sentences for French, list of padded sentences for English)
	"""
	return pad_french_corpus(french), pad_english_corpus(english)

def build_vocab(sentences):
	"""
	DO NOT CHANGE

  Builds vocab from list of sentences

	:param sentences:  list of sentences, each a list of words
	:return: tuple of (dictionary: word --> unique index, pad_token_idx)
  """
	tokens = []
	for s in sentences: tokens.extend(s)
	all_words = sorted(list(set([STOP_TOKEN,PAD_TOKEN,UNK_TOKEN,FIRST_SPECIAL,LAST_SPECIAL] + tokens)))

	vocab =  {word:i for i,word in enumerate(all_words)}

	return vocab

def convert_to_id(vocab, sentences):
	"""
	DO NOT CHANGE

  Convert sentences to indexed 

	:param vocab:  dictionary, word --> unique index
	:param sentences:  list of lists of words, each representing padded sentence
	:return: list of list of integers, with each row representing the word indices in the corresponding sentences
  """
	return np.stack([[vocab[word] if word in vocab else vocab[UNK_TOKEN] for word in sentence] for sentence in sentences])

	
def read_data(file_name):
	"""
	DO NOT CHANGE

  Load text data from file

	:param file_name:  string, name of data file
	:return: list of sentences, each a list of words split on whitespace
  """
	text = []
	with open(file_name, 'rt', encoding='latin') as data_file:
		for line in data_file: text.append(line.split())
	# for line in data_file: text.append([w for w in line.split() if w.isalpha()])
	return text



def get_part1_data(data_file, vocab=None):
	"""
	Read and parse the data file line by line, then tokenize the sentences.
	Create a vocabulary dictionary that maps all the unique tokens to a unique integer value.
	Then vectorize the data based on the vocabulary dictionary.

	Note we have provided most of the code.

	:param data_file: Path to the data file.
	:return: Tuple of (1-d list or array with training words in vectorized/id form), vocabulary (Dict containg index->word mapping)
	"""

	# load and concatenate data from file.
	data = read_data(data_file)

	# removes trailing STOP word that is on every sentence
	data = [x[:-1] for x in data]

	# pads/trims each sentence to exactly WINDOW_SIZE words
	data = pad_english_corpus(data)

	if vocab is None:
		vocab = build_vocab(data)

	# read in and tokenize data
	data_tokens = convert_to_id(vocab, data)

	return data_tokens, vocab

def get_part2_data(french_training_file, english_training_file, french_vocab=None, english_vocab=None):
	"""
	Use the helper functions in this file to read and parse training and test data, then pad the corpus.
	Then vectorize your train and test data based on your vocabulary dictionaries.

	:param french_training_file: Path to the french training file.
	:param english_training_file: Path to the english training file.
	:param french_test_file: Path to the french test file.
	:param english_test_file: Path to the english test file.
	
	:return: Tuple of train containing:
	(2-d list or array with english training sentences in vectorized/id form [num_sentences x 15] ),
	(2-d list or array with english test sentences in vectorized/id form [num_sentences x 15]),
	(2-d list or array with french training sentences in vectorized/id form [num_sentences x 14]),
	(2-d list or array with french test sentences in vectorized/id form [num_sentences x 14]),
	english vocab (Dict containg word->index mapping),
	french vocab (Dict containg word->index mapping),
	english padding ID (the ID used for *PAD* in the English vocab. This will be used for masking loss)
	"""
	# MAKE SURE YOU RETURN SOMETHING IN THIS PARTICULAR ORDER: train_english, test_english, train_french, test_french, english_vocab, french_vocab, eng_padding_index


	# Read English and French Data for training and testing (see read_data)
	french_data = read_data(french_training_file)
	english_data = read_data(english_training_file)

	# Pad training data (see pad_corpus)
	french_data = pad_french_corpus(french_data)
	english_data = pad_english_corpus(english_data)

	# Build vocabs
	if french_vocab is None and english_vocab is None:
		french_vocab = build_vocab(french_data)
		english_vocab = build_vocab(english_data)

	# Convert sentences to list of IDS
	french_ids = convert_to_id(french_vocab, french_data)
	english_ids = convert_to_id(english_vocab, english_data)

	return french_ids, english_ids, french_vocab, english_vocab


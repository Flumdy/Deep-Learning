import os
import numpy as np
import tensorflow as tf
import numpy as np
from preprocess import *
from rnn_model import RNN_Part2
import sys
from tqdm import tqdm

def debug_output(inputs, outputs, labels, input_vocab, output_vocab, prefix="debug"):
	print(prefix + " output:")
	decoded_symbols = tf.argmax(input=outputs, axis=2)
	i = np.random.randint(0, len(labels))
	print("#input:"+ " ".join(input_vocab[x] for x in inputs[i]))
	print("#output:"+ " ".join(output_vocab[x] for x in decoded_symbols[i][:-1].numpy()))
	print("#actual:"+" ".join(output_vocab[x] for x in labels[i][1:]))

def train(model, train_french, train_english, optimizer, batch_size=128, input_vocab=None, output_vocab=None, debug=True, test=None):
	"""
	Runs through one epoch - all training examples.

	:param model: the initialized model to use for forward and backward pass
	:param train_french: french train data (all data for training) of shape (num_sentences, WINDOW_SIZE)
	:param train_english: english train data (all data for training) of shape (num_sentences, WINDOW_SIZE)
	:return: None
	"""

	total_loss = 0
	i = 0
	p = range(0, len(train_french), batch_size)
	for start in p:
		inputs = train_french[start:start + batch_size]
		labels = train_english[start:start + batch_size]

		with tf.GradientTape() as tape:
			outputs = model.call(inputs, labels)
			l = model.loss_function(outputs, labels)
			gradients = tape.gradient(l, model.trainable_variables)
			current_loss = tf.reduce_sum(l)
			total_loss += current_loss
		i += 1

		optimizer.apply_gradients(zip(gradients, model.trainable_variables))

		if debug:
			debug_output(inputs, outputs, labels, input_vocab, output_vocab)
		if test:
			test(i)
	return total_loss / i

def test(model, test_french, test_english, batch_size=128, input_vocab=None, output_vocab=None, debug=False):
	"""
	Runs through one epoch - all testing examples.

	:param model: the initialized model to use for forward and backward pass
	:param test_french: french test data (all data for testing) of shape (num_sentences, WINDOW_SIZE)
	:param test_english: english test data (all data for testing) of shape (num_sentences, WINDOW_SIZE)
	"""

	total_loss = 0
	i = 0
	p = range(0, len(test_french), batch_size)
	for start in p:
		inputs = test_french[start:start + batch_size]
		labels = test_english[start:start + batch_size]

		outputs = model.call(inputs, labels)
		l = model.loss_function(outputs, labels)
		s = tf.reduce_sum(l)
		# can be NaN if all of the output labels are special *UNK* or similar and they don't get counted
		if not np.isnan(s):
			total_loss += s
			i += 1
		if debug:
			debug_output(inputs, outputs, labels, input_vocab, output_vocab, prefix="test")

	print(f'test loss {total_loss / i:.3f} perplexity {np.exp(total_loss / i):.3f}')


def translate_sentence(input_ids, model, output_vocab, known_output_ids=None):
	"""
    Takes a model and vocab and encodes the input_ids and generates the most likely
    output for the decoder's next word from the language model's distribution

    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

	text = []
	output_ids = [[output_vocab[START_TOKEN]]]
	reverse_vocab = {idx: word for word, idx in output_vocab.items()}
	input_array = np.array([input_ids])

	stopped = False
	i = 0
	while i < WINDOW_SIZE and not stopped:
		logits = model.call(input_array, np.array(output_ids))
		# grab the prediction for the last word
		out_index = np.argmax(np.array(logits[0][-1]))
		text.append(reverse_vocab[out_index])
		output_ids[0].append(out_index)
		i += 1
		stopped = out_index == output_vocab[STOP_TOKEN]

	print(" ".join(text))
	if output_vocab:
		print("#"+" ".join([reverse_vocab[word] for word in known_output_ids]))


def main():	

      print("Part 2 Running preprocessing...")
      #load large training set to get the vocabulary
      data_path = os.path.expanduser('/content/gdrive/My Drive/Deep Learning/Assignment3')
      english_file_1 = os.path.join(data_path, 'data/els.txt')
      english_file_2 = os.path.join(data_path, 'data/els_short.txt')
      english_file_3 = os.path.join(data_path, 'data/elt.txt')
      french_file_1 = os.path.join(data_path, 'data/fls.txt')
      french_file_2 = os.path.join(data_path, 'data/fls_short.txt')
      french_file_3 = os.path.join(data_path, 'data/flt.txt')

      train_french, train_english, french_vocab, english_vocab = get_part2_data(french_file_1, english_file_1)
      train_short_french, train_short_english, _, _ = get_part2_data(french_file_2, english_file_2, french_vocab, english_vocab)
      test_french, test_english, _, _ = get_part2_data(french_file_3, english_file_3, french_vocab, english_vocab)

      french_reverse_vocab = {idx: word for word, idx in french_vocab.items()}
      english_reverse_vocab = {idx: word for word, idx in english_vocab.items()}

      print("Preprocessing complete.")

      model = RNN_Part2(french_vocab, english_vocab)
      optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
      p = tqdm(range(21))
      print("# test 1")

      # this should take roughly 5 minutes
      for epoch in p:
        loss = train(model, train_short_french, train_short_english, optimizer,
              input_vocab=french_reverse_vocab, output_vocab=english_reverse_vocab, debug=False)
        p.set_description(f'Epoch {epoch} training loss {loss:.3f}')
        if epoch % 10 == 0:
          test(model, train_short_french, train_short_english,
            input_vocab=french_reverse_vocab, output_vocab=english_reverse_vocab, debug=True)

      print("# test 2")
      # note this may take 30-60 minutes or more
      def test_while_training(i):
        if i % 100 == 0:
          test(model, test_french, test_english,
            input_vocab=french_reverse_vocab, output_vocab=english_reverse_vocab, debug=False)
      train(model, train_french, train_english, optimizer,
          input_vocab=french_reverse_vocab, output_vocab=english_reverse_vocab, debug=True, test=test_while_training)

if __name__ == '__main__':
   main()



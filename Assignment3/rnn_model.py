import numpy as np
import tensorflow as tf
from preprocess import *

class RNN_Part1(tf.keras.Model):
	def __init__(self, vocab):
		"""
        The RNN_Part1 class predicts the next words in a sequence.
        Feel free to initialize any variables that you find necessary in the constructor.

        :param vocab_size: The number of unique words in the data
        """
		super(RNN_Part1, self).__init__()
		self.input_size =  256
		self.output_size = 128
		self.vocab = vocab
		
		self.embedded_layer = tf.keras.layers.Embedding(len(self.vocab), self.output_size)
		self.rnn_layer = tf.keras.layers.LSTM(self.input_size, return_sequences=True)
		self.dense_layer= tf.keras.layers.Dense(len(self.vocab), activation=tf.keras.activations.softmax)

		
	def call(self, inputs):
		"""
        - You must use an embedding layer as the first layer of your network 
        - You must use an LSTM or GRU as the next layer.

        :param inputs: word ids of shape (batch_size, window_size)
        :return: the batch element probabilities as a tensor
        """
		# TODO: implement the forward pass calls on your tf.keras.layers!

		embedded_output = self.embedded_layer(inputs)
		rnn_output = self.rnn_layer(embedded_output)
		dense_output = self.dense_layer(rnn_output)
		
		return dense_output

	def loss(self, probs, labels):
		"""
        Calculates average cross entropy sequence to sequence loss of the prediction

        :param probs: a matrix of shape (batch_size, window_size, vocab_size) as a tensor
        :param labels: matrix of shape (batch_size, window_size) containing the labels
        :return: the loss of the model as a tensor of size 1
        """

		mask = tf.less(labels, self.vocab[FIRST_SPECIAL]) | tf.greater(labels, self.vocab[LAST_SPECIAL])
		loss = tf.boolean_mask(tf.keras.losses.sparse_categorical_crossentropy(labels, probs), mask)
		loss = tf.math.reduce_mean(loss)
		return loss

class RNN_Part2(tf.keras.Model):
	def __init__(self, french_vocab, english_vocab):

		super(RNN_Part2, self).__init__()

		self.input_size =  256
		self.output_size = 128	
		self.french_vocab = french_vocab
		self.english_vocab = english_vocab		

		self.french_embedded_layer = tf.keras.layers.Embedding(len(self.french_vocab), self.output_size)
		self.french_rnn_layer = tf.keras.layers.LSTM(self.input_size, return_state=True)		
		self.english_embedded_layer = tf.keras.layers.Embedding(len(self.english_vocab), self.output_size)		
		self.english_rnn_layer = tf.keras.layers.LSTM(self.input_size, return_sequences=True)		
		self.dense_layer = tf.keras.layers.Dense(len(self.english_vocab), activation=tf.keras.activations.softmax)		
		
		
	def call(self, encoder_input, decoder_input):
		"""
		:param encoder_input: batched ids corresponding to french sentences
		:param decoder_input: batched ids corresponding to english sentences
		:return prbs: The 3d probabilities as a tensor, [batch_size x window_size x english_vocab_size]
		"""
		# TODO: implement the forward pass calls on your tf.keras.layers!
		# Note 1: in the diagram there are two inputs to the decoder
		#  (the decoder_input and the hidden state of the encoder)
		#  Be careful because we don't actually need the predictive output
		#   of the encoder -- only its hidden state
		# Note 2: If you use an LSTM, the hidden_state will be the last two
		#   outputs of calling the rnn. If you use a GRU, it will just be the
		#   second output.

		french_embedded_output = self.french_embedded_layer(encoder_input)
		french_rnn_output = self.french_rnn_layer(french_embedded_output)[1:]
		english_embedded_output = self.english_embedded_layer(decoder_input)
		english_rnn_output = self.english_rnn_layer(english_embedded_output, initial_state=french_rnn_output)
		decoder_output = self.dense_layer(english_rnn_output)
		
		return decoder_output

	def loss_function(self, probs, labels):
          """
          Calculates the model cross-entropy loss after one forward pass.

          :param probs:  float tensor, word prediction probabilities [batch_size x window_size x english_vocab_size]
          :param labels:  integer tensor, word prediction labels [batch_size x window_size]
          :return: the loss of the model as a tensor
          """
          labels = labels[:, :-1]
          probs = probs[:, 1:, :]			
          
          english_mask = tf.math.not_equal(labels, self.english_vocab[PAD_TOKEN])
          
          labels = tf.boolean_mask(labels, english_mask)
          probs = tf.boolean_mask(probs, english_mask)
            
          loss = tf.keras.losses.sparse_categorical_crossentropy(labels, probs)
            
          return tf.math.reduce_mean(loss)		

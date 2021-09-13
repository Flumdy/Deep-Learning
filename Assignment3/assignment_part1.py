import tensorflow as tf
import numpy as np
from preprocess import *
from rnn_model import RNN_Part1
from tqdm import tqdm
import os

def debug_output(outputs, labels, output_vocab, prefix="debug"):
	print(prefix+" output:")
	decoded_symbols = tf.argmax(input=outputs, axis=2)
	i = np.random.randint(0, len(labels))
	print("#output:"+ " ".join(output_vocab[x] for x in decoded_symbols[i].numpy()))
	print("#actual:"+" ".join(output_vocab[x] for x in labels[i]))

def train(model, train_inputs, train_labels, optimizer, batch_size=128, output_vocab=None, debug=True):
    """
    Runs through one epoch - all training examples.
    
    :param model: the initialized model to use for forward and backward pass
    :param train_inputs: train inputs (all inputs for training) of shape (num_inputs,)
    :param train_labels: train labels (all labels for training) of shape (num_labels,)
    :return: None
    """
    total_loss = 0
    i = 0
    p = range(0, len(train_inputs), batch_size)
    if debug and len(p) > 1:
        p = tqdm(p)
    for start in p:
        inputs = train_inputs[start:start + batch_size]
        labels = train_labels[start:start + batch_size]

        with tf.GradientTape() as tape:
            outputs = model.call(inputs)
            l = model.loss(outputs, labels)

            gradients = tape.gradient(l, model.trainable_variables)
            current_loss = tf.reduce_sum(l)
            total_loss += current_loss
        i += 1

        optimizer.apply_gradients(zip(gradients, model.trainable_variables))
        if isinstance(p,tqdm):
            p.set_description(f'loss {current_loss:.3f}')
        if debug:
            debug_output(outputs, labels, output_vocab)

    return total_loss/i



def test(model, test_inputs, test_labels, output_vocab=None, debug=False):
    """
    Runs through one epoch - all testing examples
    
    :param model: the trained model to use for prediction
    :param test_inputs: train inputs (all inputs for testing) of shape (num_inputs,)
    :param test_labels: train labels (all labels for testing) of shape (num_labels,)
    :returns: perplexity of the test set

    Note: perplexity is exp(total_loss/number of predictions)

    """
    total_loss = 0
    i = 0

    batch_size = 128
    for start in range(0, len(test_inputs), batch_size):
        inputs = test_inputs[start:start + batch_size]
        labels = test_labels[start:start + batch_size]

        outputs = model.call(inputs)
        l = model.loss(outputs, labels)
        s = tf.reduce_sum(l)
        # can be NaN if all of the output labels are special *UNK* or similar and they don't get counted
        if not np.isnan(s):
            total_loss += s
            i += 1
        if debug:
            debug_output(outputs, labels, output_vocab, prefix="test")

    print(f'test loss {total_loss/i:.3f} perplexity {np.exp(total_loss/i):.3f}')

def generate_sentence(words_strings, length, vocab,model):
    """
    Takes a model, vocab, selects from the most likely next word from the model's distribution

    This is only for your own exploration. What do the sequences your RNN generates look like?
    
    :param model: trained RNN model
    :param vocab: dictionary, word to id mapping
    :return: None
    """

    reverse_vocab = {idx:word for word, idx in vocab.items()}
    words = words_strings.split(' ')
    next_input = [[vocab[word] for word in words]]
    text = []+words

    for i in range(length-len(words)):
        array_input = np.array(next_input)
        probs = model.call(array_input)
        # grab the prediction for the last word
        out_index = np.argmax(np.array(probs[0][-1]))
        text.append(reverse_vocab[out_index])
        next_input[0].append(out_index)

    print(" ".join(text))


def test_jackjill(data_path):
    file_path = os.path.join(data_path, 'data/jj.txt')
    data, vocab = get_part1_data(file_path)
    train_y = data[:,1:]
    train_x = data[:,:-1]
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    model = RNN_Part1(vocab)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    p = tqdm(range(100))
    for epoch in p:
        loss = train(model, train_x, train_y, optimizer, output_vocab=reverse_vocab, debug=False)
        p.set_description(f'Epoch {epoch} training loss {loss:.3f}')

    generate_sentence("Jack and", 7, vocab, model)
    generate_sentence("To", 6, vocab, model)
    generate_sentence("Jack fell", 7, vocab, model)
    generate_sentence("And Jill", 5, vocab, model)

    test(model, train_x, train_y)

def test_short(data_path):
    file_path = os.path.join(data_path, 'data/short.txt')
    data, vocab = get_part1_data(file_path)
    testdata, _ = get_part1_data(file_path, vocab)
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    train_x = data[:, :-1]
    train_y = data[:, 1:]

    model = RNN_Part1(vocab)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epoch in range(100):
        loss = train(model, train_x, train_y, optimizer, output_vocab=reverse_vocab)
        # Print out perplexity
        print(f'Epoch {epoch} training loss {loss:.3f}')
        test(model, testdata[:, :-1], testdata[:, 1:], output_vocab=reverse_vocab, debug=True)



def test_long(data_path):
    train_file_path = os.path.join(data_path, 'data/train.txt')
    data, vocab = get_part1_data(train_file_path)
    test_file_path = os.path.join(data_path, 'data/test.txt')
    testdata, _ = get_part1_data(test_file_path, vocab)
    reverse_vocab = {idx: word for word, idx in vocab.items()}

    train_y = data[:, 1:]
    train_x = data[:, :-1]

    model = RNN_Part1(vocab)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    for epoch in range(1):
        loss = train(model, train_x, train_y, optimizer, output_vocab=reverse_vocab)
        # Print out perplexity
        print(f'Epoch {epoch} loss {loss:.3f}')

        test(model, testdata[:, :-1], testdata[:, 1:], output_vocab=reverse_vocab, debug=True)

def main():
    data_path = os.path.expanduser('/content/gdrive/My Drive/Deep Learning/Assignment3')
    print("# test 1")
    test_jackjill(data_path)
    print("# test 2")
    test_short(data_path)
    print("# test 3")
    test_long(data_path)

    
if __name__ == '__main__':
    main()

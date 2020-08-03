import numpy as np
import gzip

def get_data(inputs_file_path, labels_file_path, num_examples):
      """
      Takes in an inputs file path and labels file path, unzips both files,
      normalizes the inputs, and returns (NumPy array of inputs, NumPy
      array of labels). Read the data of the file into a buffer and use
      np.frombuffer to turn the data into a NumPy array. Keep in mind that
      each file has a header of a certain size. This method should be called
      within the main function of the assignment.py file to get BOTH the train and
      test data. If you change this method and/or write up separate methods for
      both train and test data, we will deduct points.

      Hint: look at the writeup for sample code on using the gzip library

      :param inputs_file_path: file path for inputs, something like
      'MNIST_data/t10k-images-idx3-ubyte.gz'
      :param labels_file_path: file path for labels, something like
      'MNIST_data/t10k-labels-idx1-ubyte.gz'
      :param num_examples: used to read from the bytestream into a buffer. Rather
      than hardcoding a number to read from the bytestream, keep in mind that each image
      (example) is 28 * 28, with a header of a certain number.
      :return: NumPy array of inputs as float32 and labels as int8
      """

      # TODO: Load inputs and labels
      #open inputs file and normalize values
      image_size = 784
      
      with gzip.open(inputs_file_path, 'rb') as f: 
            inputs_content = f.read()
            output_inputs = np.frombuffer(inputs_content, count=image_size * num_examples, dtype=np.uint8, offset=16)
            output_inputs = np.float32(output_inputs) / np.float32(255)
            output_inputs.resize(num_examples, image_size)
          
      #open labels file and normalize values
      with gzip.open(labels_file_path, 'rb') as g:
            labels = g.read()    
            output_labels = np.frombuffer(labels, dtype=np.int8, count=-1, offset=8)
          
          
      
      return output_inputs, output_labels
    

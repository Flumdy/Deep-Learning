# Answer the conceptual questions here
Q1: Is there anything we need to know to get your code to work? If you did not get your code working or handed in an incomplete solution please let us know what you did complete (0-4 sentences)


Q2: Why do we normalize our pixel values between 0-1? (1-3 sentences)

Normalising the pixel values means we have the same ratio of values for every input while avoiding the large (and potentially disruptive) values that non-normalised values can produce. 

Q3: Why do we use a bias vector in our forward pass? (1-3 sentences)

The bias vector is always adde dontop of any output from the inputs. What this means is we can avoid a perceptron continously outputting zeros.

Q4: Why do we separate the functions for the gradient descent update from the calculation of the gradient in back propagation? (2-4 sentences)

Seperating these functions allows for the back_propogation algorithm to be changed (or subbed out) without the need to change or update the gradient decent portion of the algorithm. 

Q5: What are some qualities of MNIST that make it a “good” dataset for a classification problem? (2-3 sentences)

Firstly, there is a set, discrete number of classes that the network needs to place an image in. This makes it easier to create a loss function from an incorrect prediction. Second, there are less variables between inputs (e.g all greyscale inputs) which means the network does not need to be as complex.

Q6: Suppose you are an administrator of the US Postal Service in the 1990s. What positive and/or negative effects would result from deploying an MNIST-trained neural network to recognize handwritten zip codes on mail? (2-4 sentences)

You could automate the process of sorting the mail by postcodes to be delivered. This would cut out the processes the mail must go through from arrival to in-process delivery. 

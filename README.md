# Tensorflow Exercises for Prof Rumshisky's ML/NLP Class

Below are several Tensorflow exercises and links to some studying materials, for UMass students in Prof Anna Rumshisky's 
Machine Learning/Natural Language Processing Seminar. I worked with Prof Rumshisky as a research intern over the summer and 
therefore have gone through roughly the same stuff to pick up Tensorflow. And I found them kind of helpful. 

Special thanks to:
- Anna Rumshisky, my summer research advisor, for her incredible patience and guidance,
- David Donahue, my co-worker, for his **indispensable** support,
- Andrej Karparthy for his enlightening blogposts,
- Chip Huyen, for kindly opensourcing her study materials at Stanford University.


## Exercise 1 Logistic Regression on MNIST 

Just to get us started with Tensorflow, in this exercise, we will use logistic regression to classify images of digits 0 - 9 
(from the MNIST dataset). In this exercise, you will see how to use Tensorflow's unique feature "placeholder" to organize data
into mini-batches for training. You will also implement a simple model (soft-max classifier) and train it using a Tensorflow 
session. 

Moreover, you will see the power of auto-differentiation by using Tensorflow's built-in GradientDescentOptimizer (If 
you have gone through the math and written backward propogation algorithms yourself, you know how much of a chore it is to push
around various matrices in order to find the gradients. Tensorflow does this for you!) Check out the classification accuracy!

## Exercise 2 Convolutional Neural Net on MNIST

Now for an upgrade, we will get to implement a Convolutional Neural Networks (CNN) for the same problem in Exercise 1, image 
classifications. In this case, we will use Tensorflow's graph structure to build layers one on top of another (convolutional, 
pooling, relu, fully connected, etc). In addition, you will also see the use of name scope and variable scope in Tensorflow.

Together with Tensorflow's saver and checkpoints feature, they can be used to save trained models and restore them when you run
your script later. This is also a great feature of Tensorflow. Play with the hyperparameters in Step 2. Can you improve the 
classification accuracy?

## Exercise 3 Word2vec Model Training

In the field of natural language processing, one task to to find good representations for word. In the word2vec model, each 
word token is represented as a vector of fixed size (300), known as the embedding vector. Good embedding representations
need to capture some correlation among words, such as semantics, association and composition and etc (I should say that what 
constitute "good" representations for words is a very deep and unresolved area, which I do not know enough to talk about).

In this exercise, you will implement your own word2vec using the "Skip-Gram" model, which hinges upon using the center words 
to predict the surrounding words as "target words". Process_data script has been provided to generate mini-batches of data. 

Check out the following if you are interested in various models for word representation:
https://cs224d.stanford.edu/lecture_notes/LectureNotes1.pdf

## Exercise 4 Character Level RNN

Recurrent Neural Network is widely used in natural language processing and in computer vision (image caption). In a recurrent 
neural network, data is processed by the cell unit and then got fed back to the cell unit at the next time step. That way, the 
recurrent layers retain a memory of previoulsy processed data. Therefore, Recurrent Neural Networks are especially suitable for 
sequential data. 

In this exercise, you will train a character level recurrent neural network on a dataset of lots of deep learning papers
(abstracts)! During training, you will get to see samples of output from the model you trained. Tensorflow provides the 
function "dynamic_rnn", so that we do not have to write the neural network computation at each time step. (Again, if you have
written back-propogation for RNN, even if it is Matlab, you'd really appreciate this feature.)

Check out the "deep learning papers" that your model has generated :)

## More studying materials:

Udacity class on Deep Learning (This is relatively easy to get started with):<br />
https://www.udacity.com/course/deep-learning--ud730

Deep Learning on Keras and Tensorflow (Keras is a library on top of Tensorflow that is supposedly easier to start wtih):
https://github.com/leriomaggio/deep-learning-keras-tensorflow

CS 224d Deep Learning for Natural Language Processing (This is a Stanford class, where you will not only learn Tensorflow but 
also Natural Language Processing):
https://cs224d.stanford.edu/

CS 20si Tensorflow and Deep Learning Research (Unfortunately there is not video lecture, just slides, but I am really grateful
for this class and the instructor Professor Hyuen):
http://web.stanford.edu/class/cs20si/syllabus.html

Udacity's many many coding projects on Deep Learning (This is really comprehensive, including the newest stuff like Reinforcement
Learning and GAN, or Generative Adversarial Networks):
https://github.com/udacity/deep-learning

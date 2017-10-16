
from __future__ import print_function
import os
import time

import tensorflow as tf

DATA_PATH = './arvix_abstracts.txt'
HIDDEN_SIZE = 200
BATCH_SIZE = 64
NUM_STEPS = 50
SKIP_STEP = 40
TEMPRATURE = 0.7
LR = 0.003
LEN_GENERATED = 300
SAVING_DIR = './rnn/checkpoints/'

def vocab_encode(text, vocab):
    return [vocab.index(x) + 1 for x in text if x in vocab]

def vocab_decode(array, vocab):
    return ''.join([vocab[x - 1] for x in array])

def read_data(filename, vocab, window=NUM_STEPS, overlap=NUM_STEPS/2):
    for text in open(filename):
        text = vocab_encode(text, vocab)
        for start in range(0, len(text) - window, overlap):
            chunk = text[start: start + window]
            chunk += [0] * (window - len(chunk))
            yield chunk

def read_batch(stream, batch_size=BATCH_SIZE):
    batch = []
    for element in stream:
        batch.append(element)
        if len(batch) == batch_size:
            yield batch
            batch = []
    yield batch

def create_rnn(seq, hidden_size=HIDDEN_SIZE):
    # Step 1 Create the cell state using tf.contrib.rnn.GRUCell
    #################### Your Code Starts Here #############################
    cell = tf.contrib.rnn.GRUCell(hidden_size)
    #################### Your Code Ends Here #############################



    # Step 2 Create the initial state place holder using tf.placeholder_with_default
    # and using cell.zero_state
    #################### Your Code Starts Here #############################

    in_state = tf.placeholder_with_default(
            cell.zero_state(tf.shape(seq)[0], tf.float32), [None, hidden_size])
    #################### Your Code Ends Here #############################

    # this line to calculate the real length of seq
    # all seq are padded to be of the same length which is NUM_STEPS
    length = tf.reduce_sum(tf.reduce_max(tf.sign(seq), 2), 1)

    # Step 3 Use tf.nn.dynamic_rnn, which takes in (cell, sequence, length, initial state)
    # to create our rnn model. The rnn model should return both the ouput (the RNN output tensor) 
    #  and the out_state, which is the final state of the RNN
    # This link may help you: https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn
    #################### Your Code Starts Here #############################

    output, out_state = tf.nn.dynamic_rnn(cell, seq, length, in_state)

    #################### Your Code Ends Here #############################

    return output, in_state, out_state

def create_model(seq, temp, vocab, hidden=HIDDEN_SIZE):
    seq = tf.one_hot(seq, len(vocab))
    #print (tf.shape(seq)) #sfd

    # Step 4 Having defined our create_rnn function, we call it to create our model
    #################### Your Code Starts Here #############################

    output, in_state, out_state = create_rnn(seq, hidden)

    #################### Your Code Ends Here #############################

    # fully_connected is syntactic sugar for tf.matmul(w, output) + b
    # it will create w and b for us
    logits = tf.contrib.layers.fully_connected(output, len(vocab), None)

    # Step 5 Use logits[:, :-1] as our model prediction, and seq[:, 1:] as the ground truths labels
    # to create our loss function (You should see that the model prediction and the ground truths 
    # are offset by 1! Why is that?)
    #################### Your Code Starts Here #############################

    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=logits[:, :-1], labels=seq[:, 1:]))

    #################### Your Code Ends Here #############################

    # sample the next character from Maxwell-Boltzmann Distribution with temperature temp
    # it works equally well without tf.exp
    sample = tf.multinomial(tf.exp(logits[:, -1] / temp), 1)[:, 0] 
    return loss, sample, in_state, out_state

def training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state):
    saver = tf.train.Saver()
    start = time.time()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('./rnn/graphs/', sess.graph)
        sess.run(tf.global_variables_initializer())
        
        ckpt = tf.train.get_checkpoint_state(os.path.dirname('./rnn/checkpoints/'))
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        iteration = global_step.eval()
        for batch in read_batch(read_data(DATA_PATH, vocab)):
            batch_loss, _ = sess.run([loss, optimizer], {seq: batch})
            if (iteration + 1) % SKIP_STEP == 0:
                print('Iter {}. \n    Loss {}. Time {}'.format(iteration, batch_loss, time.time() - start))
                online_inference(sess, vocab, seq, sample, temp, in_state, out_state)
                start = time.time()
                if not os.path.exists(SAVING_DIR):
                    os.makedirs(SAVING_DIR) 
                saver.save(sess, SAVING_DIR, iteration)
            iteration += 1

def online_inference(sess, vocab, seq, sample, temp, in_state, out_state, seed='T'):
    """ Generate sequence one character at a time, based on the previous character
    """
    sentence = seed
    state = None
    for _ in range(LEN_GENERATED):
        batch = [vocab_encode(sentence[-1], vocab)]
        feed = {seq: batch, temp: TEMPRATURE}
        # for the first decoder step, the state is None
        if state is not None:
            feed.update({in_state: state})
        index, state = sess.run([sample, out_state], feed)
        sentence += vocab_decode(index, vocab)
    print(sentence)

def main():
    vocab = (
            " $%'()+,-./0123456789:;=?ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "\\^_abcdefghijklmnopqrstuvwxyz{|}")
    seq = tf.placeholder(tf.int32, [None, None])
    temp = tf.placeholder(tf.float32)
    loss, sample, in_state, out_state = create_model(seq, temp, vocab)
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    optimizer = tf.train.AdamOptimizer(LR).minimize(loss, global_step=global_step)
    training(vocab, seq, loss, optimizer, global_step, temp, sample, in_state, out_state)
    
if __name__ == '__main__':
    main()
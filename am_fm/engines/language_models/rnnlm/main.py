import collections
import numpy as np
import os
import tensorflow as tf
import codecs
import sentencepiece as spm
import word2vec
import pickle
from tqdm import tqdm
from tensorflow.contrib import rnn
from six.moves import range

flags = tf.app.flags
logging = tf.logging

# load sentencepiece tokenizer

flags.DEFINE_string('data_path', ".", "data directory.")
flags.DEFINE_string('dataset', "twitter", "subdirectory of data directory.")
flags.DEFINE_string('eval_data', "twitter", "subdirectory of data directory.")
flags.DEFINE_string('data_size', "10k", "training data size.")
flags.DEFINE_string('test_data', " ", "test data")
flags.DEFINE_string('model_name', "lstm_10k", "subdirectory where model is saved.")
flags.DEFINE_string('model_path', "twitter/fm_model", "subdirectory where model is saved.")
flags.DEFINE_string('embedding_name', "embeddings_10k.npy", "path to trained word2vec")
flags.DEFINE_string('tokenizer_path', "bpe_full.model", "path to sentencepiece tokenizer")
flags.DEFINE_string('bpe_vocab', "twitter/bpe_full.vocab", "path to sentencepiece tokenizer")
flags.DEFINE_string('hyp_out', 'twitter', 'folder path to save hypothesis perplexity')
flags.DEFINE_string('ref_out', 'twitter', 'folder path to save reference perplexity')
flags.DEFINE_integer('batch_size', 256, "number of samples in each batch.")
flags.DEFINE_integer('seq_len', 50, "fix the sequence length")
flags.DEFINE_integer('embedding_size', 128 , 'specify embedding dimension')
flags.DEFINE_integer('num_epochs', 251, "number of epochs to run")
flags.DEFINE_integer('decay_threshold', 5, 'learning rate decay after decay_threshold epochs')
flags.DEFINE_integer('valid_summary', 1, 'validation interval')
flags.DEFINE_boolean('use_sp', False, 'whether to use sentencepiece tokenizer')
flags.DEFINE_boolean('do_train', False, 'whether to run training procedure')
flags.DEFINE_boolean('do_eval', False, 'whether to run evaluation procedure')
flags.DEFINE_boolean('do_reload', False, 'whether to reload the latest_checkpoint')
flags.DEFINE_boolean('do_dstc_eval', False, 'whether to run computation of dstc data sentence-level perplexity')
flags.DEFINE_list('num_nodes', '64,48,32', 'hidden layer size')
flags.DEFINE_float('dropout', 0.2, 'dropout rate')
flags.DEFINE_float('learning_rate', 0.01, 'learning rate')
flags.DEFINE_float('decay_rate', 0.5,'learning rate decay rate')
flags.DEFINE_float('gradient_clipping', 5.0, 'gradient clipping')

FLAGS = flags.FLAGS

global data_list, dictionary, reverse_dictionary,vocabulary_size

if FLAGS.use_sp:
    # load sentencepiece tokenizer
    sp = spm.SentencePieceProcessor()
    sp.Load(FLAGS.tokenizer_path)
    with codecs.open(FLAGS.bpe_vocab, mode='r', encoding='utf-8') as rf:
        lines = rf.readlines()
    lines = [line.strip().split('\t')[0] for line in lines]
    dictionary = {v:k for k, v in enumerate(lines)}
    reverse_dictionary = {k:v for k, v in enumerate(lines)}
    vocabulary_size = sp.GetPieceSize()

# function for reading data from each input file
def read_data(filename):
  data = []
  with codecs.open(filename, encoding='utf-8', mode='r') as f:
    for line in f.readlines():
        if line.strip():
            if FLAGS.use_sp:
                tokenized_line = sp.EncodeAsPieces(line.strip())
            else:
                tokenized_line = line.strip().split()
            tokenized_line.insert(0, '<s>')
            tokenized_line.append('</s>')
            data.extend(tokenized_line)
  return data


def build_dataset_sp(documents):
    chars = []
    # This is going to be a list of lists
    # Where the outer list denote each document
    # and the inner lists denote words in a given document
    data_list = []
    
    # Traverse through all the text we have
    # to replace each string word with the ID of the word
    for d in documents:
        data = list()
        for char in d:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            index = sp.PieceToId(char)
            data.append(index)
            
        data_list.append(data)
    return data_list


# function for converting tokenized list to idx

def build_dataset(documents, model_path):
    chars = []
    # This is going to be a list of lists
    # Where the outer list denote each document
    # and the inner lists denote words in a given document
    data_list = []
    
    # dump_vocabulary
    if os.path.exists(os.path.join(model_path, 'word2id.dict')):
        dictionary = pickle.load(open(os.path.join(model_path, 'word2id.dict'), 'rb'))

    else:
        for d in documents:
            chars.extend(d)
        print('%d Words found.'%len(chars))
        count = []
        # Get the word sorted by their frequency (Highest comes first)
        count.extend(collections.Counter(chars).most_common())
        # Create an ID for each word by giving the current length of the dictionary
        # And adding that item to the dictionary
        # Start with 'UNK' that is assigned to too rare words
        dictionary = dict({'<s>':0, '</s>':1, '<unk>':2, '<pad>':3})
        for char, c in count:
            # Only add a bigram to dictionary if its frequency is more than 10
            if c > 10:
                dictionary[char] = len(dictionary)    
        pickle.dump(dictionary, open(os.path.join(model_path, 'word2id.dict'), 'wb'))

    unk_count = 0
    # Traverse through all the text we have
    # to replace each string word with the ID of the word
    for d in documents:
        data = list()
        for char in d:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if char in dictionary:
                index = dictionary[char]        
            else:
                index = dictionary['<unk>']
                unk_count += 1
            data.append(index)
            
        data_list.append(data)
    print('%d unk present.'%unk_count)  
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data_list, dictionary, reverse_dictionary

def learn_w2v(num_files, data_list, reverse_dictionary, embedding_size, vocabulary_size, embed_name, data_len):
    ## CBOW: Learning Word Vectors
    word2vec.define_data_and_hyperparameters(num_files, data_list, reverse_dictionary, embedding_size, vocabulary_size) 
    word2vec.print_some_batches()
    word2vec.define_word2vec_tensorflow()

    # We save the resulting embeddings as embeddings-tmp.npy 
    # If you want to use this embedding for the following steps
    # please change the name to embeddings.npy and replace the existing
    word2vec.run_word2vec(embed_name, data_len)


class DataGeneratorSeq(object):
    
    def __init__(self,text,batch_size,num_unroll):
        # Text where a bigram is denoted by its ID
        self._text = text
        # Number of bigrams in the text
        self._text_size = len(self._text)
        # Number of datapoints in a batch of data
        self._batch_size = batch_size
        # Num unroll is the number of steps we unroll the RNN in a single training step
        # This relates to the truncated backpropagation we discuss in Chapter 6 text
        self._num_unroll = num_unroll
        # We break the text in to several segments and the batch of data is sampled by
        # sampling a single item from a single segment
        self._segments = self._text_size//self._batch_size

        self._cursor = [offset * self._segments for offset in range(self._batch_size)]
        
    def next_batch(self):
        '''
        Generates a single batch of data
        '''
        # Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
        batch_data = np.zeros((self._batch_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size),dtype=np.float32)
        
        # Fill in the batch datapoint by datapoint
        for b in range(self._batch_size):
            # If the cursor of a given segment exceeds the segment length
            # we reset the cursor back to the beginning of that segment
            if self._cursor[b]+1>=self._text_size:
                self._cursor[b] = b * self._segments
            
            # Add the text at the cursor as the input
            batch_data[b] = self._text[self._cursor[b]]
            # Add the preceding word as the label to be predicted
            batch_labels[b]= self._text[self._cursor[b]+1]                      
            # Update the cursor
            self._cursor[b] = (self._cursor[b]+1)%self._text_size
                    
        return batch_data,batch_labels
        
    def unroll_batches(self):
        '''
        This produces a list of num_unroll batches
        as required by a single step of training of the RNN
        '''
        unroll_data,unroll_labels = [],[]
        for ui in range(self._num_unroll):
            data, labels = self.next_batch()            
            unroll_data.append(data)
            unroll_labels.append(labels)
        
        return unroll_data, unroll_labels
    
    def reset_indices(self):
        '''
        Used to reset all the cursors if needed
        '''
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]

    def get_segment_size(self):
        return self._segments // self._num_unroll


# Learning rate decay logic

def decay_learning_rate(session, v_perplexity):
    global decay_threshold, decay_count, min_perplexity  
    # Decay learning rate
    if v_perplexity < min_perplexity:
      decay_count = 0
      min_perplexity= v_perplexity
    else:
      decay_count += 1

    if decay_count >= decay_threshold:
      print('\t Reducing learning rate')
      decay_count = 0
      session.run(inc_gstep)

if __name__=="__main__":

    num_files = 2

    dir_name = os.path.join(FLAGS.data_path, FLAGS.dataset) 

#    filenames = ['train_clean_{}.txt'.format(FLAGS.data_size), 'valid_clean.txt', 'test_clean.txt']
    filenames = ['train', 'valid']

    #model_path = os.path.join(dir_name, FLAGS.model_name)
    model_path = FLAGS.model_path
    if not os.path.exists(model_path):
        os.mkdir(model_path)

    # ## Reading data
    # Data will be stored in a list of lists where the each list represents a document and document is a list of words. We will then break the text into words.

    global documents

    documents = []

    for i in range(num_files):    
        print('\nProcessing file %s'%os.path.join(dir_name, filenames[i]))
        
        words = read_data(os.path.join(dir_name, filenames[i]))
        
        documents.append(words)
        print('Data size (Characters) (Document %d) %d' %(i,len(words)))
        print('Sample string (Document %d) %s'%(i,words[:50]))


    # ## Building the Dictionaries
    # Builds the following. To understand each of these elements, let us also assume the text "I like to go to school"
    # 
    # * `dictionary`: maps a string word to an ID (e.g. {I:0, like:1, to:2, go:3, school:4})
    # * `reverse_dictionary`: maps an ID to a string word (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
    # * `count`: List of list of (word, frequency) elements (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
    # * `data` : Contain the string of text we read, where string words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
    # 
    # It also introduces an additional special token `UNK` to denote rare words to are too rare to make use of.

    if not FLAGS.use_sp:
	# print some statistics about data
        data_list, dictionary, reverse_dictionary = build_dataset(documents, model_path)
        print('Sample data', data_list[0][:10])
        print('Sample data', data_list[1][:10])
        print('Vocabulary: ', len(dictionary))
        vocabulary_size = len(dictionary)
    else:
        data_list = build_dataset_sp(documents)
        print('Sample data', data_list[0][:10])
        print('Sample data', data_list[1][:10])
        vocabulary_size = sp.GetPieceSize()
    del documents  # To reduce memory.


    # data_len = max([len(data) for data in data_list])


    # Train or load word2vec embeddings
    embedding_name = os.path.join(model_path, FLAGS.embedding_name)
    learn_w2v(1, [data_list[0]], reverse_dictionary, FLAGS.embedding_size, vocabulary_size, embedding_name, len(data_list[0]))


    # =========================================================
    # Define Graph

    # Number of neurons in the hidden state variables
    num_nodes = [int(node) for node in FLAGS.num_nodes]

    # ## Defining Inputs and Outputs
    # 
    # In the code we define two different types of inputs. 
    # * Training inputs (The stories we downloaded) (batch_size > 1 with unrolling)
    # * Validation inputs (An unseen validation dataset) (bach_size =1, no unrolling)
    # * Test inputs (New story we are going to generate) (batch_size=1, no unrolling)

    tf.reset_default_graph()

    # Training Input data.
    train_inputs, train_labels = [],[]
    train_labels_ohe = []
    # Defining unrolled training inputs
    for ui in range(FLAGS.seq_len):
        train_inputs.append(tf.placeholder(tf.int32, shape=[FLAGS.batch_size],name='train_inputs_%d'%ui))
        train_labels.append(tf.placeholder(tf.int32, shape=[FLAGS.batch_size], name = 'train_labels_%d'%ui))
        train_labels_ohe.append(tf.one_hot(train_labels[ui], vocabulary_size))
        
    # Validation data placeholders
    valid_inputs = tf.placeholder(tf.int32, shape=[1],name='valid_inputs')
    valid_labels = tf.placeholder(tf.int32, shape=[1], name = 'valid_labels')
    valid_labels_ohe = tf.one_hot(valid_labels, vocabulary_size)

    # Text generation: batch 1, no unrolling.
    test_input = tf.placeholder(tf.int32, shape=[1],name='test_input')


    # ## Loading Word Embeddings to TensorFlow
    # We load the previously learned and stored embeddings to TensorFlow and define tensors to hold embeddings
    embed_mat = np.load(embedding_name)
    embeddings_size = embed_mat.shape[1]

    embed_init = tf.constant(embed_mat)
    embeddings = tf.Variable(embed_init, name='embeddings')


    # Defining embedding lookup operations for all the unrolled
    # trianing inputs
    train_inputs_embeds = []
    for ui in range(FLAGS.seq_len):
        # We use expand_dims to add an additional axis
        # As this is needed later for LSTM cell computation
        train_inputs_embeds.append(tf.expand_dims(tf.nn.embedding_lookup(embeddings, train_inputs[ui]),0))

    # Defining embedding lookup for operations for all the validation data
    valid_inputs_embeds = tf.nn.embedding_lookup(embeddings,valid_inputs)

    # Defining embedding lookup for operations for all the testing data
    test_input_embeds = tf.nn.embedding_lookup(embeddings, test_input)

    # ## Defining Model Parameters
    # 
    # Now we define model parameters. Compared to RNNs, LSTMs have a large number of parameters. Each gate (input, forget, memory and output) has three different sets of parameters.

    print('Defining softmax weights and biases')
    # Softmax Classifier weights and biases.
    w = tf.Variable(tf.truncated_normal([num_nodes[-1], vocabulary_size], stddev=0.01))
    b = tf.Variable(tf.random_uniform([vocabulary_size],0.0,0.01))

    print('Defining the LSTM cell')
    # Defining a deep LSTM from Tensorflow RNN API

    # First we define a list of LSTM cells
    # num_nodes here is a sequence of hidden layer sizes
    cells = [tf.nn.rnn_cell.LSTMCell(n) for n in num_nodes]

    # We now define a dropout wrapper for each LSTM cell
    dropout_cells = [
        rnn.DropoutWrapper(
            cell=lstm, input_keep_prob=1.0,
            output_keep_prob=1.0-FLAGS.dropout, state_keep_prob=1.0,
            variational_recurrent=True, 
            input_size=tf.TensorShape([embeddings_size]),
            dtype=tf.float32
        ) for lstm in cells
    ]

    # We first define a MultiRNNCell Object that uses the 
    # Dropout wrapper (for training)
    stacked_dropout_cell = tf.nn.rnn_cell.MultiRNNCell(dropout_cells)
    # Here we define a MultiRNNCell that does not use dropout
    # Validation and Testing
    stacked_cell = tf.nn.rnn_cell.MultiRNNCell(cells)


    # ## Defining LSTM Computations
    # Here first we define the LSTM cell computations as a consice function. Then we use this function to define training and test-time inference logic.

    print('LSTM calculations for unrolled inputs and outputs')
    # =========================================================
    # Training inference logic

    # Initial state of the LSTM memory.
    initial_state = stacked_dropout_cell.zero_state(FLAGS.batch_size, dtype=tf.float32)

    # Defining the LSTM cell computations (training)
    train_outputs, initial_state = tf.nn.dynamic_rnn(
        stacked_dropout_cell, tf.concat(train_inputs_embeds,axis=0), 
        time_major=True, initial_state=initial_state
    )

    # Reshape the final outputs to [seq_len*batch_size, num_nodes]
    final_output = tf.reshape(train_outputs,[-1,num_nodes[-1]])

    # Computing logits
    logits = tf.matmul(final_output, w) + b
    # Computing predictions
    train_prediction = tf.nn.softmax(logits)

    # Reshape logits to time-major fashion [seq_len, batch_size, vocabulary_size]
    time_major_train_logits = tf.reshape(logits,[FLAGS.seq_len, FLAGS.batch_size,-1])

    # We create train labels in a time major fashion [seq_len, batch_size, vocabulary_size]
    # so that this could be used with the loss function
    time_major_train_labels = tf.reshape(tf.concat(train_labels,axis=0),[FLAGS.seq_len, FLAGS.batch_size])

    # Perplexity related operation
    train_perplexity_without_exp = tf.reduce_sum(tf.concat(train_labels_ohe,0)*-tf.log(train_prediction+1e-10))/(FLAGS.seq_len*FLAGS.batch_size)

    # =========================================================
    # Validation inference logic

    # Separate state for validation data
    initial_valid_state = stacked_cell.zero_state(1, dtype=tf.float32)

    # Validation input related LSTM computation
    valid_outputs, initial_valid_state = tf.nn.dynamic_rnn(
        stacked_cell, tf.expand_dims(valid_inputs_embeds,0), 
        time_major=True, initial_state=initial_valid_state
    )

    # Reshape the final outputs to [1, num_nodes]
    final_valid_output = tf.reshape(valid_outputs,[-1,num_nodes[-1]])

    # Computing logits
    valid_logits = tf.matmul(final_valid_output, w) + b
    # Computing predictions
    valid_prediction = tf.nn.softmax(valid_logits)

    # Perplexity related operation
    valid_perplexity_without_exp = tf.reduce_sum(valid_labels_ohe*-tf.log(valid_prediction+1e-10))


    # ## Calculating LSTM Loss
    # We calculate the training loss of the LSTM here. It's a typical cross entropy loss calculated over all the scores we obtained for training data (`loss`) and averaged and summed in a specific way.

    # We use the sequence-to-sequence loss function to define the loss
    # We calculate the average across the batches
    # But get the sum across the sequence length
    loss = tf.contrib.seq2seq.sequence_loss(
        logits = tf.transpose(time_major_train_logits,[1,0,2]),
        targets = tf.transpose(time_major_train_labels),
        weights= tf.ones([FLAGS.batch_size, FLAGS.seq_len], dtype=tf.float32),
        average_across_timesteps=False,
        average_across_batch=True
    )

    loss = tf.reduce_sum(loss)


    # ## Defining Learning Rate and the Optimizer with Gradient Clipping
    # Here we define the learning rate and the optimizer we're going to use. We will be using the Adam optimizer as it is one of the best optimizers out there. Furthermore we use gradient clipping to prevent any gradient explosions.

    # In[30]:


    # Used for decaying learning rate
    gstep = tf.Variable(0, trainable=False)

    # Running this operation will cause the value of gstep
    # to increase, while in turn reducing the learning rate
    inc_gstep = tf.assign(gstep, gstep+1)

    # Adam Optimizer. And gradient clipping.
    tf_learning_rate = tf.train.exponential_decay(0.001, gstep, decay_steps=1, decay_rate=FLAGS.decay_rate)

    print('Defining optimizer')
    optimizer = tf.train.AdamOptimizer(tf_learning_rate)
    gradients, v = zip(*optimizer.compute_gradients(loss))
    gradients, _ = tf.clip_by_global_norm(gradients, FLAGS.gradient_clipping)
    optimizer = optimizer.apply_gradients(
        zip(gradients, v))

    inc_gstep = tf.assign(gstep, gstep+1)

    # ### Learning rate Decay Logic
    # 
    # Here we define the logic to decrease learning rate whenever the validation perplexity does not decrease
    # Learning rate decay related
    # If valid perpelxity does not decrease
    # continuously for this many epochs
    # decrease the learning rate
    decay_threshold = FLAGS.decay_threshold
    # Keep counting perplexity increases
    decay_count = 0
    min_perplexity = 1e10
    best_perplexity = 1e10

    # ### Running Training, Validation and Generation
    # 
    # We traing the LSTM on existing training data, check the validaiton perplexity on an unseen chunk of text and generate a fresh segment of text
    train_perplexity_ot = []
    valid_perplexity_ot = []

    session = tf.InteractiveSession()
    saver = tf.train.Saver()
    tf.global_variables_initializer().run()
    print('Initialized')

    train_gen = DataGeneratorSeq(data_list[0], FLAGS.batch_size, FLAGS.seq_len)
    # Defining the validation data generator
    valid_gen = DataGeneratorSeq(data_list[1],1,1)
    test_gen =  DataGeneratorSeq(data_list[1],1,1)
    if FLAGS.test_data != '':
        test_gen =  DataGeneratorSeq(FLAGS.test_data,1,1)
    else:
        test_gen =  DataGeneratorSeq(data_list[1],1,1)


    train_steps_per_document = train_gen.get_segment_size()
    valid_steps_per_document = valid_gen.get_segment_size()
    test_steps_per_document = test_gen.get_segment_size()

    feed_dict = {}
    average_loss = 0
    # =========================================================
    # Training Procedure
    if FLAGS.do_train:
        if FLAGS.do_reload:
            saver.restore(session, tf.train.latest_checkpoint(model_path))
        for step in range(FLAGS.num_epochs):
           print('Training (Epoch: %d)'%step)        
           for doc_step_id in tqdm(range(train_steps_per_document)):
               u_data, u_labels = train_gen.unroll_batches()
               for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                   feed_dict[train_inputs[ui]] = dat
                   feed_dict[train_labels[ui]] = lbl
               feed_dict.update({tf_learning_rate:FLAGS.learning_rate})
               _, l, step_perplexity = session.run([optimizer, loss, train_perplexity_without_exp], 
                                                          feed_dict=feed_dict)
               average_loss += step_perplexity
           print('')    
           
           if (step+1) % FLAGS.valid_summary == 0:
             
               average_loss = average_loss / (train_steps_per_document*FLAGS.valid_summary)
               # The average loss is an estimate of the loss over the last 2000 batches.
               print('Average loss at step %d: %f' % (step+1, average_loss))
               print('\tPerplexity at step %d: %f' %(step+1, np.exp(average_loss)))
               train_perplexity_ot.append(np.exp(average_loss))
               average_loss = 0 # reset loss
               
               valid_loss = 0 # reset loss
                 
               # calculate valid perplexity
               # Remember we process things as bigrams
               # So need to divide by 2
               for v_step in tqdm(range(valid_steps_per_document)):
                   uvalid_data,uvalid_labels = valid_gen.unroll_batches()        
    
                   # Run validation phase related TensorFlow operations       
                   v_perp = session.run(
                       valid_perplexity_without_exp,
                       feed_dict = {valid_inputs:uvalid_data[0],valid_labels: uvalid_labels[0]}
                   )
    
                   valid_loss += v_perp
                     
                   # Reset validation data generator cursor
                   valid_gen.reset_indices() 
               print('')      
               v_perplexity = np.exp(valid_loss/valid_steps_per_document)
               print("Valid Perplexity: %.2f\n"%v_perplexity)
               valid_perplexity_ot.append(v_perplexity)
        
               if v_perplexity < best_perplexity:
                   saver.save(session, os.path.join(model_path, 'model-{:.2f}'.format(v_perplexity)))
                   best_perplexity = v_perplexity
                   
               decay_learning_rate(session, v_perplexity)
    
    if FLAGS.do_eval:    
        # load best model
        saver.restore(session, tf.train.latest_checkpoint(model_path))
        # evaluate test set
        # calculate test perplexity
        test_loss = 0 # reset loss
        for t_step in tqdm(range(test_steps_per_document)):
            utest_data,utest_labels = test_gen.unroll_batches()        

            # Run validation phase related TensorFlow operations       
            t_perp = session.run(
                valid_perplexity_without_exp,
                feed_dict = {valid_inputs:utest_data[0],valid_labels:utest_labels[0]}
            )

            test_loss += t_perp
                      
            # Reset validation data generator cursor
            test_gen.reset_indices() 
        print('')      
        t_perplexity = np.exp(test_loss/test_steps_per_document)
        print("test Perplexity: %.2f\n"%t_perplexity)

    if FLAGS.do_dstc_eval:
        # load hypothesis and references
        hyp_list = read_data(os.path.join(FLAGS.eval_data, 'hypothesis.txt'))
        splitted_hyp = []
        temp = []
        for item in hyp_list:
            temp.append(item)
            if item == '</s>':
                splitted_hyp.append(temp)
                temp = []
        print('total length of splitted hypothesis: {}'.format(len(splitted_hyp)))
        splitted_hyp = [[dictionary[token] if token in dictionary.keys() else dictionary['<unk>'] for token in item] for
                        item in tqdm(splitted_hyp)]


        ref_list = read_data(os.path.join(FLAGS.eval_data, 'reference.txt'))
        splitted_ref = []
        temp = []
        for item in ref_list:
            temp.append(item)
            if item == '</s>':
                splitted_ref.append(temp)
                temp = []
        print('total length of splitted references: {}'.format(len(splitted_ref)))
        splitted_ref = [[dictionary[token] if token in dictionary.keys() else dictionary['<unk>'] for token in item] for
                        item in tqdm(splitted_ref)]

        hyp_perplexity_ot = []
        ref_perplexity_ot = []

        print('Restoring model from {}'.format(model_path))

        # load best model
        saver.restore(session, tf.train.latest_checkpoint(model_path))

        for i, item in enumerate(tqdm(splitted_ref)):
            utest_data = item[:-1]
            utest_labels = item[1:]
            assert len(utest_data) == len(utest_labels)
            sent_perplexity = 0
            for j in range(len(utest_data)):
                # Run validation phase related TensorFlow operations
                t_perp = session.run(
                    valid_perplexity_without_exp,
                    feed_dict={valid_inputs: [utest_data[j] * 1.0], valid_labels: [utest_labels[j] * 1.0]}
                )
                sent_perplexity += t_perp
            t_perplexity = np.exp(sent_perplexity / len(utest_data))
            ref_perplexity_ot.append(t_perplexity)

        with codecs.open(FLAGS.ref_out, mode='w') as f:
            f.truncate()
        for item in ref_perplexity_ot:
            with codecs.open(FLAGS.ref_out, mode='a') as f:
                f.write(str(item) + '\n')

        print('Done writing reference perplexity file to {}'.format(os.path.join(model_path, FLAGS.ref_out)))

        # evaluate hypothesis set
        # calculate hypothesis perplexity
        for i, item in enumerate(tqdm(splitted_hyp)):
            utest_data = item[:-1]
            utest_labels = item[1:]
            assert len(utest_data) == len(utest_labels)
            sent_perplexity = 0
            for j in range(len(utest_data)):
                # Run validation phase related TensorFlow operations
                t_perp = session.run(
                    valid_perplexity_without_exp,
                    feed_dict={valid_inputs: [utest_data[j] * 1.0], valid_labels: [utest_labels[j] * 1.0]}
                )
                sent_perplexity += t_perp
            t_perplexity = np.exp(sent_perplexity / len(utest_data))
            hyp_perplexity_ot.append(t_perplexity)

        with codecs.open(FLAGS.hyp_out, mode='w') as f:
            f.truncate()
        for item in hyp_perplexity_ot:
            with codecs.open(FLAGS.hyp_out, mode='a') as f:
                f.write(str(item) + '\n')

        print('Done writing hypothesis perplexity file to {}'.format(os.path.join(model_path, FLAGS.hyp_out)))

    session.close()

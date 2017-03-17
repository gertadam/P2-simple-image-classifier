
# coding: utf-8

# # Image Classification
# In this project, you'll classify images from the [CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar.html).  The dataset consists of airplanes, dogs, cats, and other objects. You'll preprocess the images, then train a convolutional neural network on all the samples. The images need to be normalized and the labels need to be one-hot encoded.  You'll get to apply what you learned and build a convolutional, max pooling, dropout, and fully connected layers.  At the end, you'll get to see your neural network's predictions on the sample images.
# ## Get the Data
# Run the following cell to download the [CIFAR-10 dataset for python](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz).

# ## Global methods & vars

# In[98]:

get_ipython().system('pip install tqdm')

import os
import time
import logging
import helper
import numpy as np
import tensorflow as tf
import pickle
import helper
import random

from datetime import timedelta
from urllib.request import urlretrieve
from os.path import isfile, isdir
from tqdm import tqdm

import problem_unittests as tests
import tarfile

#image_shape = [32, 32, 3]

def  split(start_time):
    # method for messuring the relative time usage 
    split_time = time.time()
    time_dif   = split_time - start_time
    #time_str = str(timedelta(seconds = int(round(time_dif))))
    time_str = str(timedelta(seconds = int(time_dif)))
    return time_str

def init_logging():
    log_dir = 'dlnd-log/'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_path = os.path.join(log_dir, 'event.log')
    # remember to set level=logging.CRITICAL
    logging.basicConfig(filename=log_path, format='%(levelname)s: - %(asctime)s : - %(message)s', level=logging.CRITICAL)
    logging.info('log start')  # information about when the log is "restarted"

def get_datafiles():
    cifar10_dataset_folder_path = 'cifar-10-batches-py'

    # Use Floyd's cifar-10 dataset if present\n",
    floyd_cifar10_location = '/input/cifar-10/python.tar.gz'

    if isfile(floyd_cifar10_location):
        tar_gz_path = floyd_cifar10_location
    else:
        tar_gz_path = 'cifar-10-python.tar.gz'
    
    class DLProgress(tqdm):
        last_block = 0

        def hook(self, block_num=1, block_size=1, total_size=None):
            self.total = total_size
            self.update((block_num - self.last_block) * block_size)
            self.last_block = block_num

    if not isfile(tar_gz_path):
    if not isfile('cifar-10-python.tar.gz'):
        with DLProgress(unit='B', unit_scale=True, miniters=1, desc='CIFAR-10 Dataset') as pbar:
            urlretrieve(
                'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz',
                'cifar-10-python.tar.gz',
                pbar.hook)
    
    if not isdir(cifar10_dataset_folder_path):
        with tarfile.open('cifar-10-python.tar.gz') as tar:
            tar.extractall()
            tar.close()

    logging.debug("Finished looking for data files")

def get_valids():
    # Preprocess Training, Validation, and Testing Data
    helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)
    logging.debug("helper.preprocess_and_save_data(cifar10_dataset_folder_path, normalize, one_hot_encode)")
    
    # Load the Preprocessed Validation data
    logging.debug("[valid_features, valid_labels] ready for hyper parametre optimize")
    valid_features, valid_labels = pickle.load(open('preprocess_validation.p', mode='rb'))
    return    valid_features, valid_labels


def normalize(image_data):

    x = image_data.astype('float32')
    if (x.max() > 1.0 and x.min() >= 0):
        x /= 255.0
    logging.debug("Data normalized")
    return x


def one_hot_encode(x):
    
    # : x: List of sample Labels
    # : return: Numpy array of one-hot encoded labels
    
    lenghts=(len(x))
    result = np.zeros((lenghts, 10))
    temp = 0
    for i in range(lenghts):    
        temp = int(x[i])
        # print (x[i])
        if temp < 0:
            x[i] = 0
            print('Beware: Negative value was found, changed to zero')
        elif temp > 9:
            x[i] = 9
            print('Beware: A large value in the data, was changed to MAX (9)')
        
        result[i][x[i]] = 1
    logging.debug("Data one-hotted")
    return result

def neural_net_image_input(image_shape):
    # : image_shape: Shape of the images
    # : return: Tensor for image input.
    
    logging.debug("neural_net_image_input")
    return tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], image_shape[2]], name='x')


def neural_net_label_input(n_classes):
    : n_classes: Number of classes
    : return: Tensor for label input.
    
    logging.debug("neural_net_label_input")
    return tf.placeholder(dtype=tf.float32, shape=[None, n_classes], name="y")

def neural_net_keep_prob_input():
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    
    logging.debug("neural_net_keep_prob_input")
    return tf.placeholder(dtype=tf.float32, name="keep_prob")

def weight_variable(shape_weights):
    initial = tf.truncated_normal(shape_weights, stddev=0.04)
    return tf.Variable(initial)

def bias_variable(shape_b):
    initial = tf.constant(0.1, shape=shape_b)
    return tf.Variable(initial)

def conv2d(x_tensor, filter_weight, stride, c_padding):
    return tf.nn.conv2d(x_tensor, filter_weight, stride, c_padding)

def max_pool(x_tensor, pool_patch_sh, pool_stride_sh, mp_padding):
    return tf.nn.max_pool(x_tensor, pool_patch_sh, pool_stride_sh, mp_padding)
    
def conv2d_maxpool(x_tensor, conv_num_outp, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outp: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    logging.debug("conv2d_maxpool(x_tensor, conv_num_outp, conv_ksize, conv_strides, pool_ksize, pool_strides)")

    tensor_shape  = x_tensor.get_shape().as_list()
    logging.debug("tensor_shape  = x_tensor.get_shape().as_list():%s", tensor_shape)
    shape_weights = [conv_ksize[0], conv_ksize[1], tensor_shape[3], conv_num_outp]
    filter_weight = weight_variable(shape_weights)  
        
    filter_bias   = bias_variable([conv_num_outp])
        
    stride       = [1, conv_strides[0], conv_strides[1],1]
    c_padding    = 'SAME'
    conv_out     = conv2d(x_tensor, filter_weight, stride, c_padding)
    bias_add_out = tf.nn.bias_add(conv_out, filter_bias)
    relu_out     = tf.nn.relu(bias_add_out)
    
    pool_patch_sh  = [1, pool_ksize[0],   pool_ksize[1], 1]
    pool_stride_sh = [1, pool_strides[0], pool_strides[1], 1]
    mp_padding     = 'VALID'
 
    return max_pool(relu_out, pool_patch_sh, pool_stride_sh, mp_padding)

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    logging.debug("flatten(x_tensor)")
    tensor_shape = x_tensor.get_shape().as_list();
    logging.debug("tensor_shape  = x_tensor.get_shape().as_list():%s", tensor_shape)
    dim = int(tensor_shape[1] * tensor_shape[2] * tensor_shape[3])
    logging.debug("dim(%s) = int(%s*%s*%s)", dim, tensor_shape[1], tensor_shape[2], tensor_shape[3])
    logging.debug("return tf.reshape(x_tensor, [-1, dim])")
    return tf.reshape(x_tensor, [-1, dim])

def tensor_weight_bias(x_tensor, num_outputs):
    tensor_shape = x_tensor.get_shape().as_list();
    logging.debug("tensor_shape  = x_tensor.get_shape().as_list():%s", tensor_shape)
    num_input    = tensor_shape[1]
    
    
    weight_fc1 = weight_variable([num_input, num_outputs])
    biases_fc1 = bias_variable([num_outputs])

    mm_weight = tf.matmul(x_tensor, weight_fc1)
    mm_w_n_b  = tf.nn.bias_add(mm_weight, biases_fc1)
    return mm_w_n_b

def fully_conn(x_tensor, num_outputs):
    """
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    logging.debug("fully_conn(x_tensor, num_outputs)")
    
    mm_w_n_b = tensor_weight_bias(x_tensor, num_outputs)
    
    logging.debug("fully_conn ended - return tf.nn.relu(mm_w_n_b)")
  
    return tf.nn.relu(mm_w_n_b)

def output(x_tensor, num_outputs):
    """
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    logging.debug("output - return fully_conn(x_tensor, num_outputs)")
    return tensor_weight_bias(x_tensor, num_outputs)

def conv_net(x, keep_prob,):
    """
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
        
    logging.debug("conv_net(x, keep_prob,) - start")

    model = x
    model_shape = model.get_shape().as_list();
    logging.debug("model_shape  = model.get_shape().as_list():%s", model_shape)

    num_conv_layers       =  4
    outputs               = 15          # *n -> 15, 30, 45, 60 
    ck                    =  3          # +1 ->  3,  4,  5,  6
    conv_strides = (1,1)
    pool_ksize   = (2,2)
    pool_strides = (2,2)
    
    for n in range (1, num_conv_layers):
        conv_ksize   = (ck,ck)
        model = conv2d_maxpool(model, (outputs*n), conv_ksize, conv_strides, pool_ksize, pool_strides)
        model_shape = model.get_shape().as_list();
        logging.debug("model_shape  = model.get_shape().as_list():%s", model_shape)
        model = tf.nn.local_response_normalization(model)
        ck += 1
    
    model = flatten(model)
    model_shape = model.get_shape().as_list();
    logging.debug("model_shape  = model.get_shape().as_list():%s", model_shape)
        
    model = fully_conn(model, 28)
    model_shape = model.get_shape().as_list();
    logging.debug("model_shape  = model.get_shape().as_list():%s", model_shape)
    model = tf.nn.dropout(model, keep_prob)
    
    model = fully_conn(model, 14)
    model_shape = model.get_shape().as_list();
    logging.debug("model_shape  = model.get_shape().as_list():%s", model_shape)
    model = tf.nn.dropout(model, keep_prob)

    num_outputs = 10
    logging.debug("def conv_net(x, keep_prob,) - return output(model, num_outputs)")
    
    return output(model, num_outputs)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    logging.debug("def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch)")
    logging.debug("session.run(optimizer,feed_dict={x:feature_batch,y:label_batch,keep_prob:keep_probability})")
    return session.run(optimizer, feed_dict={   x: feature_batch,
                                                y: label_batch,
                                                keep_prob: keep_probability })

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    train_accu = session.run(accuracy, feed_dict={ keep_prob: 1., x:feature_batch, y:label_batch })
    train_loss = session.run(cost,     feed_dict={ keep_prob: 1., x:feature_batch, y:label_batch })
    
    # I propose instead, just to use the validation data for setting a break on "no improvement"
    # when we're probably not likely to see anymore fall in loss (or increase in accuracy) 
    # assuming the Nets performace must have plateaued (or the loss might even have increased).

    global lowest_loss      # lowest loss measured on a valid batch so far
    global best_epoch       # to keep score of the best 
    
    global no_learning      # True if no reduction in valid loss was recorded for "a while"
    global stop             # stop running when loss has not decreased for [stop] batch-lines
    global i                # init

    valid_accu = session.run(accuracy, feed_dict={ keep_prob: 1., x:valid_features, y:valid_labels })
    valid_loss = session.run(cost,     feed_dict={ keep_prob: 1., x:valid_features, y:valid_labels })

    time_pass = ''
    time_pass = split(start_time)
    msg = "since start {0:>10} Loss-tr:{1:>7.5}, Accu-tr:{2:>7.2%} Loss-Val:{3:>7.5}, Accu-Val:{4:>7.2%} "
    logging.info(msg.format(time_pass, train_loss, train_accu, valid_loss, valid_accu))
    print       (msg.format(time_pass, train_loss, train_accu, valid_loss, valid_accu))
    
    if valid_loss < lowest_loss:
        lowest_loss = valid_loss
        best_epoch  = epoch
        i = 0
        time_pass = ''
        time_pass = split(start_time)
        msg="lowest_loss: {:>7.5} best_epoch: {:>3} i:{} "
        logging.info(msg.format(lowest_loss, best_epoch+1, i))
        print       (msg.format(lowest_loss, best_epoch+1, i))
    else:
        if (i < stop): 
            i += 1
            logging.debug("i:%s",i)
        else:
            no_learning = True

def train_single_batch():
    print('Checking the Training on a Single Batch...')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
    
        # Training cycle
        for epoch in range(epochs):
            batch_i = 1
            for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
            print('Epoch {:>3}, Batch {} '.format(epoch + 1, batch_i), end='')
            print_stats(sess, batch_features, batch_labels, cost, accuracy)
    print("finished Checking the Training on a Single Batch...!!\n\n")


def train_on_5_batches():
    print('Training...on 5')
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        
        # Training cycle
        for epoch in range(epochs):
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                print('Epoch {:>3}, Batch {} '.format(epoch + 1, batch_i), end='')
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
                if (no_learning == True):
                    break                 # no reduction in loss was recorded for "a while"
                
        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)
    print("finished Training...on 5 - Fully Train the Model!!\n\n")


def train_convergence():
    msg = "Tune Parameters:    epochs  {0:>5},  batch_size  {1:>5}, keep_probability: {2:>5.0%} "
    logging.info(msg.format(epochs, batch_size, keep_probability))
    print (msg.format(epochs, batch_size,  keep_probability))
    msg = "lowest_loss {0:>5},    stop  {1:>5},  no_learning: {2}                     i:{3}"
    logging.info(msg.format(lowest_loss, stop, no_learning, i))
    print (msg.format(lowest_loss, stop, no_learning, i))

    save_model_path = './image_classification'
    
    chp_dir = 'chapters/'
    if not os.path.exists(chp_dir):
        os.makedirs(chp_dir)
    chp_path = os.path.join(chp_dir, 'chapter')
    
    if (Manual_load == True):                   # for "manual" chapter loads
        save_model_path = "./chapter3"
        
    with tf.Session() as sess:
        # Initializing the variables
        sess.run(tf.global_variables_initializer())
        saver.restore(sess,save_model_path)
        logging.info("restored : %s", save_model_path)
        print("restored :", save_model_path)
    
        if (Manual_load == True):               # after manual loads
            save_model_path = './image_classification'
            
        # Training cycle
        print('...Continue training...')
        for epoch in range(epochs):
            saver = tf.train.Saver()
            chp_path_num = chp_path+str(epoch)
            chapter = saver.save(sess, chp_path_num)
            msg="saved {:>30}"
            logging.info(msg.format(chp_path_num))
            print(msg.format(chp_path_num))
            # Loop over all batches
            n_batches = 5
            for batch_i in range(1, n_batches + 1):
                for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
                    train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
                msg='Epoch {:>3}, Batch {} '
                print(msg.format((epoch + 1), batch_i), end='')
                logging.info(msg.format(epoch + 1, batch_i))
                print_stats(sess, batch_features, batch_labels, cost, accuracy)
                if (no_learning == True):
                    msg="no_learning=={0} - # no reduction in loss was recorded since epoch {1}"
                    logging.info(msg.format(no_learning, best_epoch))
                    print(msg.format(no_learning, best_epoch))
                    break                 # no reduction in loss was recorded for "a while"
            if (no_learning == True):
                msg="no_learning=={0} - # no reduction in loss was recorded since epoch {1}"
                logging.info(msg.format(no_learning, best_epoch))
                print(msg.format(no_learning, best_epoch))
                break                 # no reduction in loss was recorded for "a while"

        # Save Model
        saver = tf.train.Saver()
        save_path = saver.save(sess, save_model_path)
    logging.info("finished ...Continue training !!!!\n")
    print("finished ...Continue training - train_convergence !!!!\n\n")



# {MAIN}
# globals and init
best_epoch       =    -1     # init (to keep score of the best) 

epochs           =     1
batch_size       =    64
keep_probability =     0.85   # used in conjunction with dropout
lowest_loss      =   999      # initialize the global lowest loss 
stop             =    10      # stop running when loss has not decreased for [stop] batch-lines
no_learning      = False      # no reduction in valid loss was recorded for "a while"
i                =     0      # init


start_time =    time.time()
init_logging()

get_datafiles()

valid_features, valid_labels = get_valids()

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# training
# --------

train_single_batch()

# All globals remain unchanged for this run 
epochs           =     1
batch_size       =    64
keep_probability =     0.85   # used in conjunction with dropout
stop             =    10      # stop running when loss has not decreased for [stop] batch-lines
no_learning      = False      # no reduction in valid loss was recorded for "a while"
i                =     0      # init
save_model_path = './image_classification'

train_on_5_batches()


#  Tune Parameters
Manual_load      = False
epochs           =  2000
batch_size       =   500
keep_probability =     0.85   # used in conjunction with dropout
lowest_loss      =   999      # initialize the global lowest loss 
stop             =    10      # stop running when loss has not decreased for stop batch-lines
no_learning      = False      # no reduction in valid loss was recorded for "a while"
i                =     0      # init

train_convergence()



# ## Test Model


# Set batch size if not already set
try:
    if batch_size:
        pass
except NameError:
    batch_size = 64

save_model_path = './image_classification'
n_samples = 4
top_n_predictions = 3

def test_model():
    """
    Test the saved model against the test dataset
    """

    test_features, test_labels = pickle.load(open('preprocess_training.p', mode='rb'))
    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load model
        loader = tf.train.import_meta_graph(save_model_path + '.meta')
        loader.restore(sess, save_model_path)

        # Get Tensors from loaded model
        loaded_x = loaded_graph.get_tensor_by_name('x:0')
        loaded_y = loaded_graph.get_tensor_by_name('y:0')
        loaded_keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')
        
        # Get accuracy in batches for memory limitations
        test_batch_acc_total = 0
        test_batch_count = 0
        
        for train_feature_batch, train_label_batch in helper.batch_features_labels(test_features, test_labels, batch_size):
            test_batch_acc_total += sess.run(
                loaded_acc,
                feed_dict={loaded_x: train_feature_batch, loaded_y: train_label_batch, loaded_keep_prob: 1.0})
            test_batch_count += 1

        print('Testing Accuracy: {}\n'.format(test_batch_acc_total/test_batch_count))

        # Print Random Samples
        """
        # only in the notebook
        # --------------------
        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(test_features, test_labels)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels, loaded_keep_prob: 1.0})
        helper.display_image_predictions(random_test_features, random_test_labels, random_test_predictions)
        """

test_model()


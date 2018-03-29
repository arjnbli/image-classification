import tensorflow as tf
import numpy as np
import os
from scipy.misc import imread
from random import shuffle

def load_tiny_imagenet(path, dtype=np.float32):
 
 # First load wnids
  with open(os.path.join(path, 'wnids.txt'), 'r') as f:
    wnids = [x.strip() for x in f]

  # Map wnids to integer labels
  wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

  # Use words.txt to get names for each class
  with open(os.path.join(path, 'words.txt'), 'r') as f:
    wnid_to_words = dict(line.split('\t') for line in f)
    for wnid, words in wnid_to_words.items():
      wnid_to_words[wnid] = [w.strip() for w in words.split(',')]
  class_names = [wnid_to_words[wnid] for wnid in wnids]

  # Next load training data.
  X_train = []
  y_train = []
  for i, wnid in enumerate(wnids):
    if (i + 1) % 20 == 0:
      print ('loading training data for synset %d / %d' % (i + 1, len(wnids)))
    # To figure out the filenames we need to open the boxes file
    boxes_file = os.path.join(path, 'train', wnid, '%s_boxes.txt' % wnid)
    with open(boxes_file, 'r') as f:
      filenames = [x.split('\t')[0] for x in f]
    num_images = len(filenames)
    
    X_train_block = np.zeros((num_images, 64, 64, 3), dtype=dtype)
    y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int64)
    for j, img_file in enumerate(filenames):
      img_file = os.path.join(path, 'train', wnid, 'images', img_file)
      img = imread(img_file)
      if img.ndim==2:
        temp_train=np.zeros([64,64,3])
        temp_train[:,:,0]=img
        temp_train[:,:,1]=img
        temp_train[:,:,2]=img
        X_train_block[j,:,:,:]=temp_train
      else:
        X_train_block[j,:,:,:]=img 
    X_train.append(X_train_block)
    y_train.append(y_train_block)
      
  # We need to concatenate all training data
  X_train = np.concatenate(X_train, axis=0)
  y_train = np.concatenate(y_train, axis=0)
  
  # Next load validation data
  with open(os.path.join(path, 'val', 'val_annotations.txt'), 'r') as f:
    img_files = []
    val_wnids = []
    for line in f:
      img_file, wnid = line.split('\t')[:2]
      img_files.append(img_file)
      val_wnids.append(wnid)
    num_val = len(img_files)
    y_val = np.array([wnid_to_label[wnid] for wnid in val_wnids])
    X_val = np.zeros((num_val, 64,64, 3), dtype=dtype)
    for i, img_file in enumerate(img_files):
      img_file = os.path.join(path, 'val', 'images', img_file)
      img = imread(img_file)
      if img.ndim==2:
        temp_val=np.zeros([64,64,3])
        temp_val[:,:,0]=img
        temp_val[:,:,1]=img
        temp_val[:,:,2]=img
        X_val[i,:,:,:]=temp_train
      else:
        X_val[i,:,:,:] = img
      

  # Next load test images
  # Students won't have test labels, so we need to iterate over files in the
  # images directory.
  img_files = os.listdir(os.path.join(path, 'test', 'images'))
  X_test = np.zeros((len(img_files), 64, 64, 3), dtype=dtype)
  for i, img_file in enumerate(img_files):
    img_file = os.path.join(path, 'test', 'images', img_file)
    img = imread(img_file)
    if img.ndim==2:
        temp_test=np.zeros([64,64,3])
        temp_test[:,:,0]=img
        temp_test[:,:,1]=img
        temp_test[:,:,2]=img
        X_test[i,:,:,:]=temp_test
    else:
        X_test[i,:,:,:] = img
    

  y_test = None
  y_test_file = os.path.join(path, 'test', 'test_annotations.txt')
  if os.path.isfile(y_test_file):
    with open(y_test_file, 'r') as f:
      img_file_to_wnid = {}
      for line in f:
        line = line.split('\t')
        img_file_to_wnid[line[0]] = line[1]
    y_test = [wnid_to_label[img_file_to_wnid[img_file]] for img_file in img_files]
    y_test = np.array(y_test)
  
  

  return {
    'class_names': class_names,
    'X_train': X_train,
    'y_train': y_train,
    'X_val': X_val,
    'y_val': y_val,
    'X_test': X_test,
    'y_test': y_test,
    'class_names': class_names,
    }
  
def convertToOneHot(vector, num_classes=None):
    """
    Converts an input 1-D vector of integers into an output
    2-D array of one-hot vectors, where an i'th input value
    of j will set a '1' in the i'th row, j'th column of the
    output array.

    Example:
        v = np.array((1, 0, 4))
        one_hot_v = convertToOneHot(v)
        print one_hot_v

        [[0 1 0 0 0]
         [1 0 0 0 0]
         [0 0 0 0 1]]
    """

    assert isinstance(vector, np.ndarray)
    assert len(vector) > 0

    if num_classes is None:
        num_classes = np.max(vector)+1
    else:
        assert num_classes > 0
        assert num_classes >= np.max(vector)

    result = np.zeros(shape=(len(vector), num_classes))
    result[np.arange(len(vector)), vector] = 1
    return result.astype(int)

def data_shuffle(training_images,training_labels):
    """
    Shuffles numpy array/list of training images and 
    corresponding training_labels
    
    Arguments:
    training_images --Both have the same first dimension
    training_labels --other dimensions may vary
    
    Returns:
    shuffled_images -- training_images shuffled along the first dimension
    shuffled_labels -- training_labels shuffled along the first dimension

    """
    k=training_images.shape
    N=k[0]
    ind_list = [i for i in range(N)]
    shuffle(ind_list)
    shuffled_images=training_images[ind_list, :,:,:]
    shuffled_labels=training_labels[ind_list,]
    return shuffled_images,shuffled_labels



data=load_tiny_imagenet('/tin/tiny-imagenet-200')




x_t,y_t=data['X_train'],data['y_train']
x_v,y_v=data['X_val'],data['y_val']
X_train,y_train=data_shuffle(x_t,convertToOneHot(y_t))
X_val,y_val=data_shuffle(x_v,convertToOneHot(y_v))



def filter_weights(shape):
    """
    Initializes the filter weights
    
    Argument:
    shape -- list with shape [filter_height,filter_width,input_channels,output_channels]
 
    Returns:
    Tensor with the same shape as input argument and dtype "float

    """
    
    init=tf.truncated_normal(shape,stddev=0.1)
    return tf.Variable(init)

def filter_biases(shape):
    """
    Initializes the filter biases
    
    Argument:
    shape -- list with shape [filter_height,filter_width,input_channels,output_channels]
 
    Returns:
    Tensor with the same shape as input argument and  and dtype "float

    """
    init=tf.constant(0,dtype=tf.float32,shape=shape)
    return tf.Variable(init)

  
def batch_norm_scale():
    '''
    Initializes batch normalization scale term for each layer
    '''
    init=tf.constant(1.0,dtype=tf.float32)
    return tf.Variable(init)
    
def batch_norm_offset():
    '''
    Initializes batch normalization offset term for each layer
    '''
    init=tf.constant(0,dtype=tf.float32)
    return tf.Variable(init)

def convolution(inp,fil,strides,bias,scale,offset): 
    """
    Performs convolution --> RELU ---> Batch Normalization
    
    Argument:
    inp -- tensor with shape [batch_size,img_height,img_width,n_channels]
    fil -- tensor with shape [filter_height,filter_width,input_channels,output_channels]
    strides -- list, strides to be taken along each dimesnion of input image
    bias -- tensor with shape [output_channels]
    scale -- batch normalization scale term
    offset -- batch normalization offset term
    Returns:
    init_batch_normalized -- tensor
    """
    init=tf.nn.conv2d(inp,fil,strides,padding='SAME',use_cudnn_on_gpu=True)+bias
    init_mean,init_variance=tf.nn.moments(init,axes=[0,1,2])
    variance_epsilon=1e-3
    init_batch_normalized=tf.nn.batch_normalization(init,init_mean,init_variance,offset,scale,variance_epsilon)
    return tf.nn.relu(init_batch_normalized)
def create_placeholders(n_x, n_y):
    """
    Creates the placeholders for the tensorflow session.
    
    Arguments:
    n_x -- scalar, image height
    n_y -- scalar, number of classes (from 0 to 199, so -> 200)
    
    Returns:
    X -- placeholder for the data input, of shape [None,n_x,n_x,3] and dtype "float"
    Y -- placeholder for the input labels, of shape [None,n_y] and dtype "float"

    """

    X = tf.placeholder(dtype=tf.float32,shape=[None,n_x,n_x,3]) 
    Y = tf.placeholder(dtype=tf.float32,shape=[None,n_y])
      
    return X, Y

def forward_propagation(X):
    """
    Implements the forward propagation for the model
    
    Arguments:
    X -- input dataset placeholder, of shape (batch_size,image_height,image_width,image_depth)
   

    Returns:
    y_conv -- the output of the 14 layer RESNET model
    """
    #input images are randomly flipped left or right
    d1=tf.map_fn(lambda img: tf.image.random_flip_left_right(img), X)
    #this is followed by random saturation
    d2=tf.map_fn(lambda img: tf.image.random_saturation(img, 0.5, 2.0), d1)
    #random crop reduce image size from (64,64,3) to (56,56,3)
    x=tf.map_fn(lambda img: tf.random_crop(img, np.array([56, 56, 3])), d2)
     
    W_1=filter_weights([7,7,3,64])
    b_1=filter_biases([64])
    scale_1=batch_norm_scale()
    offset_1=batch_norm_offset()
    out_1=convolution(x,W_1,[1,2,2,1],b_1,scale_1,offset_1)

    W_2=filter_weights([3,3,64,64])
    b_2=filter_biases([64])
    scale_2=batch_norm_scale()
    offset_2=batch_norm_offset()
    out_2=convolution(out_1,W_2,[1,1,1,1],b_2,scale_2,offset_2)

    W_3=filter_weights([3,3,64,64])
    b_3=filter_biases([64])
    scale_3=batch_norm_scale()
    offset_3=batch_norm_offset()
    out_3=convolution(out_2,W_3,[1,1,1,1],b_3,scale_3,offset_3)

    W_4=filter_weights([3,3,64,64])
    b_4=filter_biases([64])
    scale_4=batch_norm_scale()
    offset_4=batch_norm_offset()
    in_4=(out_3+out_1)
    out_4=convolution(in_4,W_4,[1,1,1,1],b_4,scale_4,offset_4)

    W_5=filter_weights([3,3,64,64])
    b_5=filter_biases([64])
    scale_5=batch_norm_scale()
    offset_5=batch_norm_offset()
    out_5=convolution(out_4,W_5,[1,1,1,1],b_5,scale_5,offset_5)

    W_6=filter_weights([3,3,64,128])
    b_6=filter_biases([128])
    scale_6=batch_norm_scale()
    offset_6=batch_norm_offset()
    in_6=in_4+out_5
    out_6=convolution(in_6,W_6,[1,2,2,1],b_6,scale_6,offset_6)

    W_7=filter_weights([3,3,128,128])
    b_7=filter_biases([128])
    scale_7=batch_norm_scale()
    offset_7=batch_norm_offset()  
    out_7=convolution(out_6,W_7,[1,1,1,1],b_7,scale_7,offset_7)

    W_8=filter_weights([3,3,128,128])
    b_8=filter_biases([128])
    scale_8=batch_norm_scale()
    offset_8=batch_norm_offset()
    W_in_8=filter_weights([1,1,64,128])
    b_in_8=filter_biases([128])
    scale_in_8=batch_norm_scale()
    offset_in_8=batch_norm_offset()
    in_8=out_7+convolution(in_6,W_in_8,[1,2,2,1],b_in_8,scale_in_8,offset_in_8)
    out_8=convolution(in_8,W_8,[1,1,1,1],b_8,scale_8,offset_8)
 
    W_9=filter_weights([3,3,128,128])
    b_9=filter_biases([128])
    scale_9=batch_norm_scale()
    offset_9=batch_norm_offset()
    out_9=convolution(out_8,W_9,[1,1,1,1],b_9,scale_9,offset_9)

    W_10=filter_weights([3,3,128,256])
    b_10=filter_biases([256])
    scale_10=batch_norm_scale()
    offset_10=batch_norm_offset()
    in_10=in_8+out_9
    out_10=convolution(in_10,W_10,[1,2,2,1],b_10,scale_10,offset_10)
 
    W_11=filter_weights([3,3,256,256])
    b_11=filter_biases([256])
    scale_11=batch_norm_scale()
    offset_11=batch_norm_offset()
    out_11=convolution(out_10,W_11,[1,1,1,1],b_11,scale_11,offset_11)

    W_12=filter_weights([3,3,256,256])
    b_12=filter_biases([256])
    scale_12=batch_norm_scale()
    offset_12=batch_norm_offset()
    W_in_12=filter_weights([1,1,128,256])
    b_in_12=filter_biases([256])
    scale_in_12=batch_norm_scale()
    offset_in_12=batch_norm_offset()
    in_12=out_11+convolution(in_10,W_in_12,[1,2,2,1],b_in_12,scale_in_12,offset_in_12)
    out_12=convolution(in_12,W_12,[1,1,1,1],b_12,scale_12,offset_12)

    W_13=filter_weights([3,3,256,256])
    b_13=filter_biases([256])
    scale_13=batch_norm_scale()
    offset_13=batch_norm_offset()
    out_13=convolution(out_12,W_13,[1,1,1,1],b_13,scale_13,offset_13)

    conv_out=tf.reduce_mean((in_12+out_13),axis=[1,2])



    out_flat=tf.reshape(conv_out,[-1,256])
    W_fc=filter_weights([256,200])
    b_fc=filter_biases([200])
    y_conv=tf.matmul(out_flat,W_fc)+b_fc
    
    return y_conv


def compute_cost(Yhat, Y):
    """
    Computes the cost
    
    Arguments:
    Yhat -- output of forward propagation of shape
    Y -- "true" labels vector placeholder, same shape as Yhat
    
    Returns:
    cost - Tensor of the cost function
    """
    logits = Yhat
    labels = Y
    
    cost = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
       
    return cost

def model(X_train, y_train, X_val, Y_val, 
          num_iterations = 30000, batch_size = 400):
    """
    Implements a 14-layer deep residual network.
    
    Arguments:
    X_train -- training set, of shape (1000000,64,64,3)
    Y_train -- test set, of shape (1000000,200)
    X_test -- training set, of shape (10000,64,64,3)
    Y_test -- test set, of shape (10000,200)
    
    num_iterations -- number of steps of the optimization loop
    batch_size -- size of a minibatch
       
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
      
    # Placeholders of shape (n_x, n_y) created
    x_in,y_=create_placeholders(64,200)
    # Forward propagation
    y_conv = forward_propagation(x_in)
    # Cost function added to tensorflow graph
    cost = compute_cost(y_conv,y_)
    
    
    # Backpropagation:Optimizer defined
    global_step = tf.Variable(0, trainable=False)
    learning_rate=tf.train.exponential_decay(1e-2, global_step, 1000, 0.5, staircase=True)
    optimizer=tf.train.AdamOptimizer(learning_rate)
    train=optimizer.minimize(cost)
    
    #accuracy metrics defined
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    top5=tf.reduce_mean(tf.cast(tf.nn.in_top_k(predictions=y_conv, targets=tf.argmax(y_, 1), k=5), tf.float32))
    
    # Variable initializer
    init = tf.global_variables_initializer()

    # Session starts
    sess=tf.Session()
    init=tf.global_variables_initializer()
      
    #variable initializer run inside session
    sess.run(init)
        
    # Training Loop
    for i in range(num_iterations):
        if i%250==0:
            X_train,y_train=data_shuffle(X_train,y_train)
        j=np.random.randint(0,250)
        X_train_batch,y_train_batch=X_train[batch_size*j:batch_size*(j+1),:,:,:],y_train[batch_size*j:batch_size*(j+1),:]
        train_dict={x_in:X_train_batch,y_:y_train_batch}
        if i%100==0:
            train_accuracy=sess.run(accuracy,feed_dict=train_dict)
            top5_accuracy=sess.run(top5,feed_dict={x_in:X_val,y_:y_val})        
            validation_accuracy=sess.run(accuracy,feed_dict={x_in:X_val,y_:y_val})
            print('Iterations:%d   Training Accuracy:%g  Validation Accuracy:%g Top5:%g'%(i,train_accuracy,validation_accuracy,top5_accuracy))
        sess.run(train,feed_dict=train_dict)

        
        
        
model(X_train, y_train, X_val, y_val)
        
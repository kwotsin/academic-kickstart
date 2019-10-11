---
# Documentation: https://sourcethemes.com/academic/docs/managing-content/

title: "Transfer Learning"
subtitle: "`[Archive]` Transfer Learning in TensorFlow using a Pre-trained Inception-Resnet-V2 Model."
summary: "`[Archive]` Transfer Learning in TensorFlow using a Pre-trained Inception-Resnet-V2 Model."
authors: []
tags: []
categories: [archive]
date: 2017-02-11T20:16:36+01:00
lastmod: 2017-02-11T20:16:36+01:00
featured: false
draft: false

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
# Focal points: Smart, Center, TopLeft, Top, TopRight, Left, Right, BottomLeft, Bottom, BottomRight.
image:
  caption: ""
  focal_point: ""
  preview_only: false

# Projects (optional).
#   Associate this post with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `projects = ["internal-project"]` references `content/project/deep-learning/index.md`.
#   Otherwise, set `projects = []`.
projects: []
---

> **Author's note:** This article is ported over from my previous blog, is treated as archive and will be provided as is for anyone who finds the content useful. Writing the article originally served as one of the stepping stones for me to learn TensorFlow in the earlier days.

---


In this guide, we will see how we can perform transfer learning using the official pre-trained model offered by Google, which can be found in TensorFlow's [model library](https://github.com/tensorflow/models) and downloaded [here](http://download.tensorflow.org/models/inception_resnet_v2_2016_08_30.tar.gz). As I have mentioned in my previous post on [creating TFRecord files](https://kwotsin.github.io/tech/2017/01/29/tfrecords.html), one thing that I find really useful in using TensorFlow-slim over other deep learning libraries is the ready access to the best pretrained models offered by Google. This guide will build upon my previous guide on creating TFRecord files and show you how to use the inception-resnet-v2 model released by Google.

**Note:** The code was run on TensorFlow 0.12 and Python 2.7, but the GitHub code was updated to suit TensorFlow 1.01. For a smooth running of the code, you should clone the GitHub repo instead of copying the code here, since the GitHub code will reflect the updates in TensorFlow.

---

### Define Key Information
First let us import some of the important modules and libraries. The imports **`inception_preprocessing`** and **`inception_resnet_v2`** comes from two python files from the TF-slim [models library](https://github.com/tensorflow/models/tree/master/slim) which will be included in the source code later.

```python
import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging
import inception_preprocessing
from inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
import os
import time
slim = tf.contrib.slim
```

Then we will state the information about the dataset and the files we need to locate. We create a labels to name dictionary for us to know what our predictions are.

```python
#================ DATASET INFORMATION ======================
#State dataset directory where the tfrecord files are located
dataset_dir = '.'

#State where your log file is at. If it doesn't exist, create it.
log_dir = './log'

#State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

#State the image size you're resizing your images to. We will use the default inception size of 299.
image_size = 299

#State the number of classes to predict:
num_classes = 5

#State the labels file and read it
labels_file = './labels.txt'
labels = open(labels_file, 'r')

#Create a dictionary to refer each label to their string name
labels_to_name = {}
for line in labels:
    label, string_name = line.split(':')
    string_name = string_name[:-1] #Remove newline
    labels_to_name[int(label)] = string_name

#Create the file pattern of your TFRecord files so that it could be recognized later on
file_pattern = 'flowers_%s_*.tfrecord'

#Create a dictionary that will help people understand your dataset better. This is required by the Dataset class later.
items_to_descriptions = {
    'image': 'A 3-channel RGB coloured flower image that is either tulips, sunflowers, roses, dandelion, or daisy.',
    'label': 'A label that is as such -- 0:daisy, 1:dandelion, 2:roses, 3:sunflowers, 4:tulips'
}
```

Now we need to give some information about how we will train the model. I have chosen to use the number of training epochs instead of using the number of training steps as it is more intuitive to know how many times the model have seen the entire dataset. The batch_size is dependent upon your GPU memory size. If you get a resource exhausted error, one way you could fix this is by reducing your batch size. As the model is rather large, I find that with my GPU of around 3.5GB free memory, I could only fit a maximum of 10 examples per batch.

Also, because we will using an exponentially decaying learning rate that decays after a certain number of epoch, we will need some information about the decay rate we want and how many epochs to wait before decaying the learning rate to something smaller. You can change the **`num_epochs`** to a smaller value to try something out fast.

```python
#================= TRAINING INFORMATION ==================
#State the number of epochs to train
num_epochs = 70

#State your batch size
batch_size = 10

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2
```

---

### Creating a Dataset Object
Firstly, we need to define a function called **`get_split`** that allows us to obtain a specific split - training or validation - of the TFRecord files we created and load all the necessary information into a **`Dataset`** class for convenience. The required items - such as the decoder (and its two essential dictionaries which are explained later) and number of examples - are all collated into the **`Dataset`** class so that it makes it easy for us to obtain the information later on and for the **`DatasetDataProvider`** class to obtain Tensors from the dataset eventually.

We first check the arguments and create a general path to locate the TFRecord Files with the following code in the function:

```python
#First check whether the split_name is train or validation
if split_name not in ['train', 'validation']:
    raise ValueError('The split_name %s is not recognized.\
    Please input either train or validation as the split_name' % (split_name))

#Create the full path for a general file_pattern to locate the tfrecord_files
file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))
```

Now we need to count the number of examples in all of the shards of TFRecord files.

```python
#Count the total number of examples in all of these shard 
num_samples = 0
file_pattern_for_counting = 'flowers_' + split_name
tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir) if\                                                       file.startswith(file_pattern_for_counting)]
for tfrecord_file in tfrecords_to_count:
    for record in tf.python_io.tf_record_iterator(tfrecord_file):
        num_samples += 1
```

Of course, you can certainly get this value by referring back to your old code when you first created TFRecord files, which was what the original TF-slim code suggested (to know your training examples beforehand), but I find it more convenient to not refer, and you wouldn't need to change more of your code if you decide to change your TFRecord files split sizes. On my machine, this counting takes just 0.17 seconds for more than 3000 examples.

What is very important in this function are the **`keys_to_features`** and **`items_to_handlers`** dictionaries as well as the decoder, all of which are used by a **`DatasetDataProvider`** object to decode the TF-examples and make them into a Tensor object. This will be explained in detail in the next section.

Here is the full function for getting a dataset split:

```python
def get_split(split_name, dataset_dir, file_pattern=file_pattern):
    """
    Obtains the split - training or validation - to create a Dataset class for feeding the examples into a queue later
    on. This function will set up the decoder and dataset information all into one Dataset class so that you can avoid
    the brute work later on.
    
    Your file_pattern is very important in locating the files later. 

    INPUTS:
        - split_name(str): 'train' or 'validation'. Used to get the correct data split of tfrecord files
        - dataset_dir(str): the dataset directory where the tfrecord files are located
        - file_pattern(str): the file name structure of the tfrecord files in order to get the correct data

    OUTPUTS:
    - dataset (Dataset): A Dataset class object where we can read its various components for easier batch creation.
    """
    #First check whether the split_name is train or validation
    if split_name not in ['train', 'validation']:
        raise ValueError(\
        'The split_name %s is not recognized. Please input either train or validation as the split_name'\
        % (split_name))

    #Create the full path for a general file_pattern to locate the tfrecord_files
    file_pattern_path = os.path.join(dataset_dir, file_pattern % (split_name))

    #Count the total number of examples in all of these shard
    num_samples = 0
    file_pattern_for_counting = 'flowers_' + split_name
    tfrecords_to_count = [os.path.join(dataset_dir, file) for file in os.listdir(dataset_dir)\
                         if file.startswith(file_pattern_for_counting)]
    for tfrecord_file in tfrecords_to_count:
        for record in tf.python_io.tf_record_iterator(tfrecord_file):
            num_samples += 1

    #Create a reader, which must be a TFRecord reader in this case
    reader = tf.TFRecordReader

    #Create the keys_to_features dictionary for the decoder
    keys_to_features = {
      'image/encoded': tf.FixedLenFeature((), tf.string, default_value=''),
      'image/format': tf.FixedLenFeature((), tf.string, default_value='jpg'),
      'image/class/label': tf.FixedLenFeature(
          [], tf.int64, default_value=tf.zeros([], dtype=tf.int64)),
    }

    #Create the items_to_handlers dictionary for the decoder.
    items_to_handlers = {
    'image': slim.tfexample_decoder.Image(),
    'label': slim.tfexample_decoder.Tensor('image/class/label'),
    }

    #Start to create the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    #Create the labels_to_name file
    labels_to_name_dict = labels_to_name

    #Actually create the dataset
    dataset = slim.dataset.Dataset(
        data_sources = file_pattern_path,
        decoder = decoder,
        reader = reader,
        num_readers = 4,
        num_samples = num_samples,
        num_classes = num_classes,
        labels_to_name = labels_to_name_dict,
        items_to_descriptions = items_to_descriptions)

    return dataset
```

---

### Decoding the TF-Example through DatasetDataProvider
The main way we are going to obtain tensors from our dataset to load into a batch for training is through using a **`DatasetDataProvider`**, which allows us to get these tensors in just a few lines of code. However, I find it important to understand the intricacies within this class to really know what's happening under the hood and save yourself trouble from repeating certain actions like shuffling your examples (because it would have already been done!).

The **`DatasetDataProvider`** is composed of mainly two things: a **`ParallelReader`** object, and a decoder that will decode the TF-examples read by the **`ParallelReader`** into Tensor objects. To further illustrate:

- The **`ParallelReader`** object will keep on reading the TFRecord files with multiple readers and enqueue these examples into a **`tf.RandomShuffleQueue`** (which is created by default because the argument **`shuffle`** is True by default when creating the **`DatasetDataProvider`** object), and then TF-examples are dequeued singularly and passed onto the decoder for decoding.

- The decoder, which we specified when creating the **`Dataset`** object, takes in two dictionaries: **`keys_to_features`** and **`items_to_handlers`**. The first dictionary **`keys_to_features`** gives the **`ItemHandler`** object the information about each TF-example so that the handler knows what to extract and convert into a Tensor. The second dictionary **`items_to_handlers`** tells the handlers the name of the Tensor to convert into, as well as the specific **`ItemHandler`** that will find the specific information in each TF-example to create a Tensor from. For instance, the **`slim.tfexample_decoder.Image()`** handler looks for **`'image/encoded'`** and **`'image/format'`** as the keys by default in order to convert these features in a TF-example into a Tensor. 

**Note:** The keys in **`keys_to_features`** have the same names that are used in the **`dataset_utils.py`** file's **`image_to_tfexample`** function, so it is best to keep it the same. If you change the names of the keys, you would have to recreate the TFRecord files from scratch with these keys. Also, you would have to feed in the image handler arguments differently, for instance, **`slim.tfexample_decoder.Image(image_key='image_content', format_key='image_format')`** if you changed the names of **`'image/encoded'`** and **`'image/format'`** to those names.

Finally, after creating the **`DatasetDataProvider`**, which inherits properties from a **`DataProvider`** class, you will obtain an object with two important items: an **`items_to_tensors`** dictionary from which we can use a **`get`** method offered by the **`DataProvider`** to extract our labels and images, and also the number of examples **`num_samples`**. In order to use the **`get`** method, the name of the tensors which we specified in **`items_to_handlers`** will come to be useful here.

---

### Creating a Batch Loading Function
Now we want to create a function that actually loads a batch from the TFRecord files after all the decoding and whatnot. This function will give you a very nice layer of abstraction for you to focus on your model training.

As mentioned previously, we will create a **`DatasetDataProvider`** class that we will use to obtain our raw image and label in Tensor form.

```python
#First create the data_provider object
data_provider = slim.dataset_data_provider.DatasetDataProvider(
    dataset,
    common_queue_capacity = 24 + 3 * batch_size,
    common_queue_min = 24)

#Obtain the raw image using the get method
raw_image, label = data_provider.get(['image', 'label'])
```

Next, we need to preprocess the raw_image to get it into the right shape for the model inference. This step is crucial as we need image to have the same shape before we can fit all of them nicely in a 4D Tensor batch of shape **`[batch_size, height, width, num_channels]`**. The preprocessing also does additional stuff like distorted bounding boxes, flipping left and right, and color distortion. Image summaries are also included for one image which you can view in Tensorboard later on.

```python
#Perform the correct preprocessing for this image depending if it is training or evaluating
image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)
```

Now we still want to keep the raw image that is not preprocessed for the inception model so that we can display it as an image in its original form. We only do a simple reshaping so that it fits together nicely in one batch. **`tf.expand_dims`** will expand the 3D tensor from a [height, width, channels] shape to [1, height, width, channels] shape, while **`tf.squeeze`** will simply eliminate all the dimensions with the number '1', which brings the raw_image back to the same 3D shape after reshaping.

```python
#As for the raw images, we just do a simple reshape to batch it up
raw_image = tf.expand_dims(raw_image, 0)
raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
raw_image = tf.squeeze(raw_image)
```

Finally, we just create the images and labels batch, using multiple threads to dequeue the examples for training. The capacity is simply the capacity for the internal FIFO queue that exists by default when you create a **`tf.train.batch`**, and a higher capacity is recommended if you have an unpredictable data input/output. This can data I/O stability can be seen through a summary created by default on TensorBoard when you use the **`tf.train.batch`** function. We also let **`allow_smaller_final_batch`** be True to use the last few examples even if they are insufficient to make a batch.

```python
#Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch
images, raw_images, labels = tf.train.batch(
    [image, raw_image, label],
    batch_size = batch_size,
    num_threads = 4,
    capacity = 4 * batch_size,
    allow_smaller_final_batch = True)
```

Here is the full function for loading the batch:

```python
def load_batch(dataset, batch_size, height=image_size, width=image_size, is_training=True):
    '''
    Loads a batch for training.

    INPUTS:
    - dataset(Dataset): a Dataset class object that is created from the get_split function
    - batch_size(int): determines how big of a batch to train
    - height(int): the height of the image to resize to during preprocessing
    - width(int): the width of the image to resize to during preprocessing
    - is_training(bool): to determine whether to perform a training or evaluation preprocessing

    OUTPUTS:
    - images(Tensor): a Tensor of the shape (batch_size, height, width, channels) that contain one batch of images
    - labels(Tensor): the batch's labels with the shape (batch_size,) (requires one_hot_encoding).

    '''
    #First create the data_provider object
    data_provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        common_queue_capacity = 24 + 3 * batch_size,
        common_queue_min = 24)

    #Obtain the raw image using the get method
    raw_image, label = data_provider.get(['image', 'label'])

    #Perform the correct preprocessing for this image depending if it is training or evaluating
    image = inception_preprocessing.preprocess_image(raw_image, height, width, is_training)

    #As for the raw images, we just do a simple reshape to batch it up
    raw_image = tf.expand_dims(raw_image, 0)
    raw_image = tf.image.resize_nearest_neighbor(raw_image, [height, width])
    raw_image = tf.squeeze(raw_image)

    #Batch up the image by enqueing the tensors internally in a FIFO queue and dequeueing many elements with tf.train.batch.
    images, raw_images, labels = tf.train.batch(
        [image, raw_image, label],
        batch_size = batch_size,
        num_threads = 4,
        capacity = 4 * batch_size,
        allow_smaller_final_batch = True)

    return images, raw_images, labels
```
---

### Create a Graph
We will encapsulate the graph construction in a **`run`** function that we only run when called from the terminal and not when we import it. We create the log directory if it doesn't exist yet.

```python
def run():
    #Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
```

Now we are finally ready to construct the graph! We first start by setting the logging level to INFO (which gives us sufficient information for training purposes), and load our dataset.

```python
with tf.Graph().as_default() as graph:
    tf.logging.set_verbosity(tf.logging.INFO) #Set the verbosity to INFO level

    #First create the dataset and load one batch
    dataset = get_split('train', dataset_dir, file_pattern=file_pattern)
    images, _, labels = load_batch(dataset, batch_size=batch_size)
```

Now we need to do some mathematics to give information that will be useful for running our training for-loop and telling the exponentially decaying learning rate when to decay.

```python
#Know the number steps to take before decaying the learning rate and batches per epoch
num_batches_per_epoch = dataset.num_samples / batch_size
num_steps_per_epoch = num_batches_per_epoch #Because one step is one batch processed
decay_steps = int(num_epochs_before_decay * num_steps_per_epoch)
```
Now we create our model inference by importing the entire model structure offered by TF-slim. We will also use the argument scope that is provided along with the model so that certain arguments like your **`weight_decay`**, **`batch_norm_decay`** and **`batch_norm_epsilon`** are appropriately valued by default. Of course, you can experiment with these parameters!

I find it important to simply just use this model structure instead of constructing one from scratch, since we'll be less prone to mistakes and the **name scopes** for the variables provided will match exactly what the checkpoint model is expecting. If you need to change the model structure, then be sure to state whichever name scope to be excluded in the variables to restore (see code below).

```python
#Create the model inference
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(
    images,
    num_classes = dataset.num_classes,
    is_training = True)

#Define the scopes that you want to exclude for restoration
exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)
```

When you restore from the checkpoint file, there are **at least two scopes** that you must exclude if you are not training the Imagenet Dataset: the Auxiliary Logits and Logits layers. Because of the difference in the number of classes (the original number of classes is meant to be 1001), restoring the inference model variables from your checkpoint file will inevitably result in a tensor shape mismatch error.

Also, when you are training on grayscale images, you would have to remove the initial input convolutional layer, which assumes you have an RGB image with 3 channels, ***if*** you set the argument **`channels=3`** for the **`Image`** decoder in the **`get_split`** function. In total, here are the 3 scopes that you can exclude:

1. InceptionResnetV2/AuxLogits
2. InceptionResnetV2/Logits
3. InceptionResnetV2/Conv2d_1a_3x3 (Optional, for Grayscale images)

Take a look at the **`inception_resnet_v2.py`** file to know what other name scopes you can exclude.

**Note:** It is **very important** to start defining the variables you want to restore immediately after the model construction if you use **`slim.get_variables_to_restore`** since it will just grab all the variables in the graph. If you define the optimizer or other variables before this function, for instance, then you might have many more variables to restore which the checkpoint model does not have.

Next, we will perform a one-hot-encoding of our labels which will be used for the categorical cross entropy loss. While we perform one-hot-encoding for the labels, our accuracy metric will measure our predictions against the the raw labels. After defining the loss, we will need to add the regularization losses as well through the **`get_total_loss`** function.

```python
#Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
one_hot_labels = slim.one_hot_encoding(labels, dataset.num_classes)

#Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
loss = tf.losses.softmax_cross_entropy(onehot_labels = one_hot_labels, logits = logits)
total_loss = tf.losses.get_total_loss()    #obtain the regularization losses as well
```

We now create the global step variable using the **`get_or_create_global_step`** function we imported from the start. This function will get a global step variable if we created one earlier or create one if we didn't. While the supervisor we will use for training later has a global_step variable created by default, we need to create one first so that we can let the exponentially decaying learning rate use it.

The **`staircase = True`** argument in the learning rate means the learning rate will face a sudden drop instead of a gradual one, and the **`decay_steps`** just means how many global steps (i.e. training steps) to take before decaying the learning rate by the **`decay_rate`**. The rest of the arguments should be quite self-explanatory.

```python
#Create the global step for monitoring the learning_rate and training.
global_step = get_or_create_global_step()

#Define your exponentially decaying learning rate
lr = tf.train.exponential_decay(
    learning_rate = initial_learning_rate,
    global_step = global_step,
    decay_steps = decay_steps,
    decay_rate = learning_rate_decay_factor,
    staircase = True)
```

Now we could just create our optimizer but we use the decaying learning rate we created above, instead of using a fixed value. We also create the train_op using **`slim.learning.create_train_op`**, which is able to perform more functions like gradient clipping or multiplication to prevent exploding or vanishing gradients. This is done rather than simply doing an **`Optimizer.minimize`** function, which simply just combines **`compute_gradients`** and **`apply_gradients`** without any gradient processing after **`compute_gradients`**.

```python
#Now we can define the optimizer that takes on the learning rate
optimizer = tf.train.AdamOptimizer(learning_rate = lr)

#Create the train_op.
train_op = slim.learning.create_train_op(total_loss, optimizer)
```

Now we simply get the predictions through extracting the probabilities predicted from **`end_points['Predictions']`**, and perform an **`argmax`** function that returns us the index of the highest probability, which is also the class label.

We will also use a streaming accuracy metric called **`tf.contrib.metrics.streaming_accuracy`**. Using a streaming accuracy means you have an averaged accuracy for all the batches you train instead of just one batch. This is far more accurate than evaluating any one random batch. If you realize, there are two items returned back by the streaming accuracy function. The **`accuracy`** is what you send to be written as a summary but the **`accuracy_update`** is the update_ops that you actually run a session for so that **`accuracy`** gets updated properly. Finally, we can create a generic name called **`metrics_op`** that will group together multiple update_ops if you have multiple ops. Although there is only one update_op in this instance, I think it is a good habit to make a grouping.

```python
#State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
predictions = tf.argmax(end_points['Predictions'], 1)
probabilities = end_points['Predictions']
accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
metrics_op = tf.group(accuracy_update)
```

Finally, we reach this part when we can just state whatever variable or tensor we want to monitor. Using **`tf.summary.scalar`** will give us the graphs we see in many TensorBoard visualizations. Also, we can create a summary operation with **`tf.summary.merge_all()`** so that we can group together all the summary operations, including the image summaries done in preprocessing, in one operation for convenience.

```python
#Now finally create all the summaries you need to monitor and group them into one summary op.
tf.summary.scalar('losses/Total_Loss', total_loss)
tf.summary.scalar('accuracy', accuracy)
tf.summary.scalar('learning_rate', lr)
my_summary_op = tf.summary.merge_all()
```

By default, you will also have 3 more scalar summaries: one coming from the TFRecord parallel reading queue, one from the internal FIFO queue of **`tf.train.batch`**, and another one from the Supervisor that counts the time taken for each global step.

Before we start training the model, we realize there are multiple ops we have: a **`train_op`**, a **`metrics_op`**, and also a **`global_step`** variable which we need to run at each training step in order to get its current count. We can define a **`train_step`** function that takes in a session and runs all these ops together to save ourselves some trouble. Also, we can print some logging information about the training loss and time taken every step - all in one function. Note that this function is defined within the graph and not outside the graph.

```python
def train_step(sess, train_op, global_step):
    '''
    Simply runs a session for the three arguments provided and gives a logging on the time elapsed
    for each global step
    '''
    #Check the time for each sess run
    start_time = time.time()
    total_loss, global_step_count, _ = sess.run([train_op, global_step, metrics_op])
    time_elapsed = time.time() - start_time

    #Run the logging to print some results
    logging.info('global step %s: loss: %.4f (%.2f sec/step)', global_step_count, total_loss, time_elapsed)

    return total_loss, global_step_count

```
As for the summary operation, we will periodically run it in our training session later on instead of running it every step (very memory consuming).

Recall that earlier on, we have defined our variables to restore immediately after constructing our inference model. These variables are now passed onto a saver for restoring. We also define a restoring function that **must take in a session as its argument**, so that this restoring function can be run by a supervisor to effectively restore the model from the checkpoint file.

```python
#Now we create a saver function that actually restores the variables from a checkpoint file in a sess
saver = tf.train.Saver(variables_to_restore)
def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)
```

---

### Using a Supervisor for Training
We can now finally define the supervisor for a training session! This training session will be created within the context of the graph.

While it is common to use a tf.Session() to train your model, using a supervisor is especially useful when you are training your models for many days. In the event of a crash, you can safely restore your model from the original log directory you specified. On top of that, the supervisor helps you deal with standard services such as creating a summaryWriter and the initialization of your global and local variables (which will cause errors if not initialized!). For more documentation on the supervisor, you can visit [here](https://www.tensorflow.org/how_tos/supervisor/).

First define your supervisor, stating the log directory and **`init_fn`** argument. As suggested by the documentation, we don't run the summary_op automatically for large models or else the training may be much slower. Instead, we will run our own summary_op (which turns out to be the same op as the supervisor's one anyway) manually every 10 steps.

```python
#Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
sv = tf.train.Supervisor(logdir = log_dir, summary_op = None, init_fn = restore_fn)
```
Now create a **`managed_session`** using the supervisor instead of using a normal session. At the start of each epoch, show how the training has been progressing. I included some print statements on the returned values as a sanity check that the values are within what we should expect. You can exclude them if you wish.

```python
with sv.managed_session() as sess:
    for step in xrange(num_steps_per_epoch * num_epochs):
        #At the start of every epoch, show the vital information:
        if step % num_batches_per_epoch == 0:
            logging.info('Epoch %s/%s', step/num_batches_per_epoch + 1, num_epochs)
            learning_rate_value, accuracy_value = sess.run([lr, accuracy])
            logging.info('Current Learning Rate: %s', learning_rate_value)
            logging.info('Current Streaming Accuracy: %s', accuracy_value)

            # optionally, print your logits and predictions for a sanity check that things are going fine.
            logits_value, probabilities_value, predictions_value, labels_value = sess.run([logits,
                                                                                           probabilities,
                                                                                           predictions,
                                                                                           labels])
            print 'logits: \n', logits_value
            print 'Probabilities: \n', probabilities_value
            print 'predictions: \n', predictions_value
            print 'Labels:\n', labels_value
```

We will run the summary operations **and** the training step every 10 steps. We will use supervisor's global step **`sv.global_step`** instead of the **`global_step`** we defined earlier on because it will take the correct global step that we save at the end of every training (if we restore our old model from the log directory). Running **`sv.summary_computed`** will let the summaries that you have produced to be written by a **`summaryWriter`** which we would normally need to create for visualizations in TensorBoard, but this is handled for us by the supervisor.

```python
#Log the summaries every 10 step.
if step % 10 == 0:
    loss, _ = train_step(sess, train_op, sv.global_step)
    summaries = sess.run(my_summary_op)
    sv.summary_computed(sess, summaries)

#If not, simply run the training step
else:
    loss, _ = train_step(sess, train_op, sv.global_step)
```

Finally, we want to see our final loss and accuracy, before we save the model to our log directory.

```python
#We log the final training loss and accuracy
logging.info('Final Loss: %s', loss)
logging.info('Final Accuracy: %s', sess.run(accuracy))

#Once all the training has been done, save the log files and checkpoint model
logging.info('Finished training! Saving model to disk now.')
sv.saver.save(sess, sv.save_path, global_step = sv.global_step)
```

---

### Output
Every epoch you should see something like this:

```bash
INFO:tensorflow:Epoch 70/70
INFO:tensorflow:Current Learning Rate: 1.08234e-09
INFO:tensorflow:Current Streaming Accuracy: 0.967516
logits: 
[[ -3.60317206  -2.0048995   -1.54166877  -5.51874399   9.99008751]
 [  2.6816113   -5.01558399  -3.73171687  10.70773602  -4.62484932]
 [ -3.77882433   0.09495973   9.99652481  -6.72253704  -0.79133511]
 [ -1.58194041   1.32812119  -4.17847872   8.6046133   -4.89695787]
 [ -2.34122205  -4.20334673   0.11126217  -4.35670137   8.62893581]
 [  0.55233192   1.03494143  -0.26470259  -0.6579538   -0.6355111 ]
 [ 11.90597343  -2.34726739  -2.70955706  -3.99541306  -4.37633419]
 [ -1.71685469  -1.58077681  -1.77023184   9.478158    -4.34770298]
 [ -1.92078197  -0.60312212  -5.80508232  -1.22880793   7.861094  ]
 [ -3.02251935  10.14412785   1.77002311  -5.54020739  -6.17683315]]
Probabilities: 
[[  1.24886276e-06   6.17498017e-06   9.81328139e-06   1.83904646e-07
    9.99982595e-01]
 [  3.26705078e-04   1.48356492e-07   5.35652191e-07   9.99672413e-01
    2.19280750e-07]
 [  1.04090464e-06   5.00925962e-05   9.99928236e-01   5.48241736e-08
    2.06471814e-05]
 [  3.76458775e-05   6.91100548e-04   2.80578956e-06   9.99267161e-01
    1.36780307e-06]
 [  1.72038108e-05   2.67247628e-06   1.99859569e-04   2.29251805e-06
    9.99777973e-01]
 [  2.72849262e-01   4.42096889e-01   1.20528363e-01   8.13396722e-02
    8.31857845e-02]
 [  9.99998689e-01   6.45499142e-07   4.49319629e-07   1.24198152e-07
    8.48561825e-08]
 [  1.37419611e-05   1.57451432e-05   1.30276867e-05   9.99956489e-01
    9.89659839e-07]
 [  5.64442635e-05   2.10800674e-04   1.16061892e-06   1.12756134e-04
    9.99618769e-01]
 [  1.91291952e-06   9.99767125e-01   2.30712700e-04   1.54268903e-07
    8.16198025e-08]]
predictions: 
[4 3 2 3 4 1 0 3 4 1]
Labels:
[4 3 2 3 4 0 0 3 4 1]

```

After running the training overnight for 9 hours, this is what I obtained after the training is done.

```bash
INFO:tensorflow:global step 17900: loss: 0.4939 (1.71 sec/step)
INFO:tensorflow:global step 17901: loss: 0.4930 (1.71 sec/step)
INFO:tensorflow:global step 17902: loss: 0.4897 (1.70 sec/step)
INFO:tensorflow:global step 17903: loss: 0.5383 (1.71 sec/step)
INFO:tensorflow:global step 17904: loss: 0.4948 (1.70 sec/step)
INFO:tensorflow:global step 17905: loss: 0.4896 (1.70 sec/step)
INFO:tensorflow:global step 17906: loss: 0.6060 (1.73 sec/step)
INFO:tensorflow:global step 17907: loss: 0.5818 (1.73 sec/step)
INFO:tensorflow:global step 17908: loss: 0.5045 (1.74 sec/step)
INFO:tensorflow:global step 17909: loss: 0.8533 (1.73 sec/step)
INFO:tensorflow:global step 17910: loss: 0.4900 (1.73 sec/step)
INFO:tensorflow:global step 17911: loss: 0.5167 (1.77 sec/step)
INFO:tensorflow:global step 17912: loss: 0.5251 (1.70 sec/step)
INFO:tensorflow:global step 17913: loss: 0.4954 (1.73 sec/step)
INFO:tensorflow:global step 17914: loss: 0.4905 (1.74 sec/step)
INFO:tensorflow:global step 17915: loss: 0.4895 (1.95 sec/step)
INFO:tensorflow:global step 17916: loss: 0.4902 (1.77 sec/step)
INFO:tensorflow:global step 17917: loss: 0.4909 (1.73 sec/step)
INFO:tensorflow:global step 17918: loss: 0.4899 (1.73 sec/step)
INFO:tensorflow:global step 17919: loss: 0.6076 (1.74 sec/step)
INFO:tensorflow:global step 17920: loss: 0.4910 (1.73 sec/step)
INFO:tensorflow:Final Loss: 0.491015
INFO:tensorflow:Final Accuracy: 0.967712
INFO:tensorflow:Finished training! Saving model to disk now.
```

---

### TensorBoard Visualization (Training)
As can be seen in the screenshot below, the accuracy roughly levels off at around 96%.


![training_accuracy.png](images/training_accuracy.png)


As expected, the learning rate decays over time in a staircase fashion (which can be seen once you set the smoothing to 0 in TensorBoard).


![learning_rate.png](images/learning_rate.png)


We see that after around 5000 training steps, the loss remained rather stagnant, meaning to say the learning rate could no longer influence much of the loss. It could also be seen that a lower learning rate than what we initially set is more favourable over time, so it is good that we used an exponentially decaying learning rate.


![losses.png](images/losses.png)


Here are some photos of the kind of image summary you can expect for any one photo.

![image_summary.png](images/image_summary.png)

And another one from an earlier training where I experimented on the learning rate.

![image_summary_2.png](images/image_summary_2.png)


---

### Source Code (Training)
Click [here](https://github.com/kwotsin/transfer_learning_tutorial/blob/master/train_flowers.py) to visit GitHub for the full training code.

---

### Evaluating on the Validation Dataset
Now when we want to evaluate the training dataset, we cannot use the same inference model when doing the training since certain layers like Dropout would have to be deactivated when evaluating. The code for the evaluation, which I have written in a new file, is unsurprisingly similar to the one used for training, except for several key differences.

**Note:** this is not representative of the full evaluation code, which you can find below. Only key differences are mentioned.

First, on top of the libraries we previously used, we will import the **`get_split`** and **`load_batch`** functions from the training file for convenience and also the matplotlib library for visualizing our plots later.

```python
from train_flowers import get_split, load_batch
import matplotlib.pyplot as plt
plt.style.use('ggplot')
```

Because there are lesser things to compute (gradients etc.) in the evaluation process, we can use a lot more examples per batch to get a more consistent accuracy. We will also run three epochs of the evaluation just to get a more stable validation accuracy. Also, instead of using the official checkpoint file we used for training, we obtain the latest checkpoint model we trained from the log directory using **`tf.train.latest_checkpoint`**. 

```python
#State your log directory where you can retrieve your model
log_dir = './log'

#Create a new evaluation log directory to visualize the validation process
log_eval = './log_eval_test'

#State the dataset directory where the validation set is found
dataset_dir = '.'

#State the batch_size to evaluate each time, which can be a lot more than the training batch
batch_size = 36

#State the number of epochs to evaluate
num_epochs = 3

#Get the latest checkpoint files
checkpoint_file = tf.train.latest_checkpoint(log_dir)
```

Recall that for the **`load_batch`** function, there is an **`is_training`** argument that we need to set as False in order to use the evaluation preprocessing. Similarly, we need to set **`is_training`** as False when creating the inference model so that certain layers like dropout will not be activated.

```python
#Create the graph...
...
images, raw_images, labels = load_batch(dataset, batch_size = batch_size, is_training = False)
...
#Now create the inference model but set is_training=False
with slim.arg_scope(inception_resnet_v2_arg_scope()):
    logits, end_points = inception_resnet_v2(images, num_classes = dataset.num_classes, is_training = False)
```

We will restore all the scopes now instead of excluding scopes, since they are suited just for our task after our training.

```python
# #get all the variables to restore from the checkpoint file and create the saver function to restore
variables_to_restore = slim.get_variables_to_restore()
saver = tf.train.Saver(variables_to_restore)
def restore_fn(sess):
    return saver.restore(sess, checkpoint_file)
```

Because there is no train_op to create that would help us increment the global step variable every evaluation, we need to create an op for increasing the global step value.

```python
#Create the global step and an increment op for monitoring
global_step = get_or_create_global_step()
global_step_op = tf.assign(global_step, global_step + 1) #no apply_gradient method so manually increasing the global_step
```

Of course, we would also need an evaluation step function instead of a train step function.

```python
#Create a evaluation step function
def eval_step(sess, metrics_op, global_step):
    '''
    Simply takes in a session, runs the metrics op and some logging information.
    '''
    start_time = time.time()
    _, global_step_count, accuracy_value = sess.run([metrics_op, global_step_op, accuracy])
    time_elapsed = time.time() - start_time

    #Log some information
    logging.info('Global Step %s: Streaming Accuracy: %.4f (%.2f sec/step)',
    global_step_count,
    accuracy_value,
    time_elapsed)

    return accuracy_value
```

We continue to log our summaries as usual and evaluate, but at the end of all the evaluation, we will plot the first 10 raw images coming from the final batch we last processed and visually see how our model performed. This is done within the same session.

```python
#Now we want to visualize the last batch's images just to see what our model has predicted
raw_images, labels, predictions = sess.run([raw_images, labels, predictions])
for i in range(10):
    image, label, prediction = raw_images[i], labels[i], predictions[i]
    prediction_name, label_name = dataset.labels_to_name[prediction], dataset.labels_to_name[label]
    text = 'Prediction: %s \n Ground Truth: %s' %(prediction_name, label_name)
    img_plot = plt.imshow(image)

    #Set up the plot and hide axes
    plt.title(text)
    img_plot.axes.get_yaxis().set_ticks([])
    img_plot.axes.get_xaxis().set_ticks([])
    plt.show()

logging.info('Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.')
```

---

### Evaluation Output

Here's what we should roughly see during the evaluation:

```bash
INFO:tensorflow:Epoch: 3/3
INFO:tensorflow:Current Streaming Accuracy: 0.9606
INFO:tensorflow:Global Step 61: Streaming Accuracy: 0.9606 (1.58 sec/step)
INFO:tensorflow:Global Step 62: Streaming Accuracy: 0.9608 (1.59 sec/step)
INFO:tensorflow:Global Step 63: Streaming Accuracy: 0.9615 (1.59 sec/step)
INFO:tensorflow:Global Step 64: Streaming Accuracy: 0.9608 (1.60 sec/step)
INFO:tensorflow:Global Step 65: Streaming Accuracy: 0.9609 (1.59 sec/step)
INFO:tensorflow:Global Step 66: Streaming Accuracy: 0.9603 (1.60 sec/step)
INFO:tensorflow:Global Step 67: Streaming Accuracy: 0.9609 (1.58 sec/step)
INFO:tensorflow:Global Step 68: Streaming Accuracy: 0.9610 (1.60 sec/step)
INFO:tensorflow:Global Step 69: Streaming Accuracy: 0.9604 (1.60 sec/step)
INFO:tensorflow:Global Step 70: Streaming Accuracy: 0.9601 (1.60 sec/step)
INFO:tensorflow:Global Step 71: Streaming Accuracy: 0.9595 (1.60 sec/step)
INFO:tensorflow:Global Step 72: Streaming Accuracy: 0.9597 (1.59 sec/step)
INFO:tensorflow:global_step/sec: 0.608502
INFO:tensorflow:Global Step 73: Streaming Accuracy: 0.9595 (1.59 sec/step)
INFO:tensorflow:Global Step 74: Streaming Accuracy: 0.9597 (1.58 sec/step)
INFO:tensorflow:Global Step 75: Streaming Accuracy: 0.9598 (1.59 sec/step)
INFO:tensorflow:Global Step 76: Streaming Accuracy: 0.9600 (1.60 sec/step)
INFO:tensorflow:Global Step 77: Streaming Accuracy: 0.9591 (1.60 sec/step)
INFO:tensorflow:Global Step 78: Streaming Accuracy: 0.9585 (1.59 sec/step)
INFO:tensorflow:Global Step 79: Streaming Accuracy: 0.9590 (1.60 sec/step)
INFO:tensorflow:Global Step 80: Streaming Accuracy: 0.9596 (1.59 sec/step)
INFO:tensorflow:Global Step 81: Streaming Accuracy: 0.9597 (1.59 sec/step)
INFO:tensorflow:Global Step 82: Streaming Accuracy: 0.9602 (1.60 sec/step)
INFO:tensorflow:Global Step 83: Streaming Accuracy: 0.9607 (1.58 sec/step)
INFO:tensorflow:Global Step 84: Streaming Accuracy: 0.9612 (1.60 sec/step)
INFO:tensorflow:Global Step 85: Streaming Accuracy: 0.9613 (1.58 sec/step)
INFO:tensorflow:Global Step 86: Streaming Accuracy: 0.9614 (1.60 sec/step)
INFO:tensorflow:Global Step 87: Streaming Accuracy: 0.9612 (1.60 sec/step)
INFO:tensorflow:Global Step 88: Streaming Accuracy: 0.9610 (1.59 sec/step)
INFO:tensorflow:Global Step 89: Streaming Accuracy: 0.9605 (1.61 sec/step)
INFO:tensorflow:Global Step 90: Streaming Accuracy: 0.9600 (1.58 sec/step)
INFO:tensorflow:Final Streaming Accuracy: 0.9596
INFO:tensorflow:Model evaluation has completed! Visit TensorBoard for more information regarding your evaluation.
```

Also, here are the some images of the last batch we plotted out. For completeness, I run the model a few times to get a few inaccurate results to show, which are quite interesting. Looking at the incorrectly predicted photos, I can see they aren't as conventional as the rest, which makes sense if the model doesn't predict it that well.

#### Correct Predictions

![correct_pred1.png](images/correct_pred1.png)

![correct_pred2.png](images/correct_pred2.png)

![correct_pred3.png](images/correct_pred3.png)



#### Incorrect Predictions

![wrong_pred1.png](images/wrong_pred1.png)

![wrong_pred2.png](images/wrong_pred2.png)

![wrong_pred3.png](images/wrong_pred3.png)

---

### TensorBoard Visualization (Evaluation)

![validation_accuracy.png](images/validation_accuracy.png)


As we can expect, the evaluation accuracy will be slightly lower than the training accuracy (96.0% against 96.7%), but it is not too far off from the training accuracy. This means the extent of overfitting isn't that large, and the model has performed rather well.

---

### Source Code (Evaluation)
Click [here](https://github.com/kwotsin/transfer_learning_tutorial/blob/master/eval_flowers.py) to visit GitHub for the full evaluation code.

---

### Comparing to Some Baselines

As a comparison, I removed the **`init_fn`** argument in the evaluation code so that we can see how a 'clean' model would perform without any checkpoint restoration.

```bash
INFO:tensorflow:Global Step 88: Streaming Accuracy: 0.2098 (1.70 sec/step)
INFO:tensorflow:Global Step 89: Streaming Accuracy: 0.2109 (1.63 sec/step)
INFO:tensorflow:Global Step 90: Streaming Accuracy: 0.2094 (1.62 sec/step)
INFO:tensorflow:Final Streaming Accuracy: 0.2093
```
Just to be very sure that the fine-tuning was what made the predictions accurate, I simply changed the checkpoint file to the original model and excluded the same final layers scopes (as done in training) for restoration in the evaluation code to see how the model would perform without fine-tuning. I obtained the following results:

```bash
INFO:tensorflow:Global Step 84: Streaming Accuracy: 0.2095 (1.65 sec/step)
INFO:tensorflow:Global Step 85: Streaming Accuracy: 0.2097 (1.64 sec/step)
INFO:tensorflow:Global Step 86: Streaming Accuracy: 0.2101 (1.61 sec/step)
INFO:tensorflow:Global Step 87: Streaming Accuracy: 0.2099 (1.65 sec/step)
INFO:tensorflow:Global Step 88: Streaming Accuracy: 0.2114 (1.64 sec/step)
INFO:tensorflow:Global Step 89: Streaming Accuracy: 0.2102 (1.59 sec/step)
INFO:tensorflow:Global Step 90: Streaming Accuracy: 0.2094 (1.59 sec/step)
INFO:tensorflow:Final Streaming Accuracy: 0.2096
```

Surprisingly, the non-finetuned model has a similar performance to one not restored from the checkpoint at all! However, we did use a different number of classes instead of the 1001 classes originally, which means the model probably wouldn't realize it has to distinguish the right kinds of classes.  Also, looking at the images shown at the end, the key difference between these two baselines was that while the 'clean' model always produced **`tulips`** as the output, the predictions for the original model was more random and included other classes.

But what if we trained a 'clean' model instead? After training the 'clean' model without any restoration for 5 epochs, here is what I obtained:

```bash
INFO:tensorflow:global step 1013: loss: 1.9649 (2.13 sec/step)
INFO:tensorflow:global step 1014: loss: 1.9005 (2.15 sec/step)
INFO:tensorflow:global step 1015: loss: 2.0220 (2.09 sec/step)
INFO:tensorflow:global step 1016: loss: 2.4279 (2.01 sec/step)
INFO:tensorflow:global step 1017: loss: 1.7286 (2.10 sec/step)
INFO:tensorflow:global step 1018: loss: 2.0034 (2.06 sec/step)
INFO:tensorflow:global step 1019: loss: 1.7627 (2.06 sec/step)
INFO:tensorflow:global step 1020: loss: 2.3635 (2.05 sec/step)
INFO:tensorflow:global step 1021: loss: 1.9307 (2.12 sec/step)
INFO:tensorflow:global step 1022: loss: 1.4757 (2.01 sec/step)
INFO:tensorflow:global step 1023: loss: 2.2738 (2.19 sec/step)
INFO:tensorflow:global step 1024: loss: 2.3575 (2.23 sec/step)
INFO:tensorflow:Epoch 5/5
INFO:tensorflow:Current Learning Rate: 9.8e-05
INFO:tensorflow:Current Streaming Accuracy: 0.538086
```

The loss seems to hover around the value 2 although the training has been done for some time. Also, while the checkpointed model gives a performance of around 80% accuracy after 5 epochs of training, the accuracy for the trained 'clean' model remains as low as 53%. Evaluating this trained 'clean' model would probably give a lower accuracy than the training.

**The Final Verdict**: Fine-tuning a model restored from the checkpoint performs the best!

---

### Some Reflections
One thing I really think could be improved is having a greater batch size for the training, which will make each gradient update far more stable and consistent. Unfortunately, my GPU (GTX 860M) has only a memory around 4GB, which is quite insufficient for training a large batch size in a large model.

Also, I believe other hyperparameters could be further experimented. Due to a rather slow GPU, it has also become rather time-consuming to experiment hyperparameters slowly. I had experimented with several learning rates such as 0.001, 0.002, and 0.005. In the end, I decided a lower initial learning rate and a more aggressive decay could let the loss converge earlier, so I used 0.0002 as the learning rate with a decay rate of 0.7.

The exponential decaying learning rate was really useful. After training for a while, my loss kind of stagnated, but following a decay, the loss noticeably went down further. I believe this is also a way to indirectly experiment with the learning rate through exploring different values throughout the training.

Perhaps to increase the speed, one way is to also stop calling the summaries so often every 10 steps. Maybe calling summaries every 30 steps is a good interval as well. The image summary could have taken up a lot of time, and isn't quite worth it since only the last image is shown every time. Perhaps this summary could be run separately from the other scalar summaries and far less periodically.

When I reduced the image size to 200, I realized the time taken per training step almost reduced by half (around 0.7-0.8s per step), and this could be a way of speeding up the training and experiment some hyperparameters within the first few epochs. Of course, one trade off is information is lost when we resize to smaller sizes.

I also decided not to use **`slim.learning.train`**, the training function previded by TF-slim. Using **`slim.learning.train`** can be a fast way to train a model, but I find that it becomes less straight forward in customizing the training process. For instance, you might want to obtain the summaries every n steps instead of every certain amount of seconds. It is more transparent in just coding the supervisor out and running a session.

Finally, I realized writing a post like this is a great way to learn.

Credits to authors of TF-slim who have contributed this [wonderful library](https://github.com/tensorflow/models/tree/master/slim) for use by many people! A large part of this post was inspired by their source code, and I highly recommend you to carefully study related material offered by them - after all, they are few of the best engineers in the world!

---

### Source Code
You can download all the code files above from GitHub

```bash
$ git clone https://github.com/kwotsin/transfer_learning_tutorial.git
```

---

### FAQ

**Q:** How do I only choose to fine-tune certain layers instead of all the layers?

**A:** By default, if you did not specify an argument for `variables_to_train` in the function `create_train_op` (as seen in the `train_flowers.py` file, this argument is set to `None` and will train all the layers instead. If you want to fine-tune only certain layers, you have to pass a list of variable names to the `variables_to_train` argument. But you may ask, "how do I know the variable names of the model?" One simple way is to simply run this code within the graph context:

```
with tf.Graph().as_default() as graph:
    .... #after you have constructed the model in the graph etc..
    for i in tf.trainable_variables():
        print i
```

You will see the exact variable names that you can choose to fine-tune.

For more information, you should visit the [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/slim/python/slim/learning.py#L374).

**Q:** Why does my evaluation code give such a poor performance although my training seem to be fine?

**A:** This could be due to an issue of how `batch_norm` is updated during training in the newer versions of TF, although I've not have the chance to investigate this properly. However, some users have mentioned that by setting `is_training=True` back in the eval code, the model works exactly as expected. You should try this method and see if it works for you.

For more information, please see this thread: https://github.com/kwotsin/transfer_learning_tutorial/issues/11

**Q:** Why is my code trying to restore variables like **`InceptionResnetV2/Repeat_1/block17_20/Conv2d_1x1/weights/Adam_1`** when they are not found in the .ckpt file?

**A:** The code is no longer trying to restore variables from the .ckpt file, but rather, from the log directory where the checkpoint models of your previous training are stored. This error happens when you have changed the code but did not remove the previous log directory, and so the Supervisor will attempt to restore a checkpoint from your previous training, which will result in a mismatch of variables.

**Solution:** Simply remove your previous log directory and run the code again. This applies to both your training file and your evaluation file. See this [issue](https://github.com/kwotsin/transfer_learning_tutorial/issues/2) for more information.

**Q:** Why is my loss performing so poorly after I updated the loss function from slim.losses.softmax_cross_entropy to tf.losses.softmax_cross_entropy?

**A:** The position of the arguments for the one-hot-labels and the predictions have changed, resulting in the wrong loss computed. This happens if you're using an older version of the repo, but I have since updated the losses to tf.losses and accounted for the change in argument positions.

**Solution: `git pull` the master branch of the repository to get the updates.**

**Q:** Why does the evaluation code fails to restore the checkpoint variables I had trained and saved? My training works correctly but the evaluation code crashes.

**A:** There was an error in the code that mistakenly allows the saver used to restore the variables to save the model variables after the training is completed. Because we made this saver exclude some variables to be restored earlier on, these excluded variables will not be saved by this saver if we use it to save all the variables when the training to be completed. Instead, the code should have used the Supervisor's saver that exists internally to save the model variables in the end, since all trained variables will then be saved.

Usually, this does not occur if you have trained your model for more than 10 minutes, since the Supervisor's saver will save the variables every 10 minutes. However, if you end your training before 10 minutes, the wrong saver would have saved only some trained variables, and not all trained variables (which is what we want).

**Solution: `git pull` the master branch of the repository to get the updates.** I have changed the training code to make the supervisor save the variables at the end of the training instead. 
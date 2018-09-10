import tensorflow as tf
import numpy as np
import argparse
from PIL import Image

parser = argparse.ArgumentParser(description='PyTorch Fashion-MNIST Training')

parser.add_argument('-d', '--model-dir', default='pretrained', type=str, metavar='PATH',
                    help='path to save checkpoints')
parser.add_argument('-i', '--input-img', type=str, help='input image for inference')
args = parser.parse_args()

BATCH_SIZE = 250
NUM_ITERATIONS = 24

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def model_fn(features, labels, mode):
    """Model function for Estimator"""
   
    # MNIST images are 28x28 pixels, and have one color channel
    input_layer = tf.reshape(features["x"], [-1, 28, 28, 1])
      
    conv1 = tf.layers.conv2d(input_layer, 64, (3, 3), padding="same", activation=tf.nn.elu)
    pool1 = tf.layers.max_pooling2d(conv1, pool_size=[2, 2], strides=2)

    conv2 = tf.layers.conv2d(pool1, 64, (3, 3), padding="same", activation=tf.nn.elu)
    pool2 = tf.layers.max_pooling2d(conv2, pool_size=[2, 2], strides=2)
    
    # Flatten tensor into a batch of vectors
    pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

    dense = tf.layers.dense(pool2_flat, 1024, activation=tf.nn.elu)
    logits = tf.layers.dense(dense, 10)
    
    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1)
    }
    
    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'predict': tf.estimator.export.PredictOutput(predictions)
            })

    # Calculate Loss (for both TRAIN and EVAL modes)
    onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
    loss = tf.losses.softmax_cross_entropy(
        onehot_labels=onehot_labels, logits=logits)
        
    tf.summary.scalar('train_loss', loss)
    tf.summary.scalar('train_accuracy', tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])[1])
    tf.summary.scalar('train_precision', tf.metrics.precision(labels=labels, predictions=predictions["classes"])[1])

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer()
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        'accuracy_eval': tf.metrics.accuracy(labels=labels, predictions=predictions["classes"]),
        'precision_eval': tf.metrics.precision(labels=labels, predictions=predictions["classes"]) 
    }

    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

if __name__ == "__main__":

    checkpointing_config = tf.estimator.RunConfig(
        save_checkpoints_secs=5 * 60,  # Save checkpoints every 5 minutes.
        keep_checkpoint_max=2,  # Retain the 2 most recent checkpoints.
    )

    mnist_classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=args.model_dir, config=checkpointing_config)

    if args.input_img:
        img = Image.open(args.input_img).convert('L')
        img = np.asarray(img.getdata(), dtype=np.float32).reshape(-1, 28, 28, 1)
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": img}, shuffle=False)
        for pred in mnist_classifier.predict(input_fn=predict_input_fn):
            print("Predicted class: {}".format(class_names[pred["classes"]]))
    else:

        from tensorflow.examples.tutorials.mnist import input_data

        mnist = input_data.read_data_sets('data', one_hot=False, validation_size=0)
        train_data = mnist.train.images  # Returns np.array
        train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
        eval_data = mnist.test.images  # Returns np.array
        eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data}, y=train_labels, batch_size=BATCH_SIZE, num_epochs=None, shuffle=True)
        # Evaluate the model and print results
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            num_epochs=1,
            shuffle=False)

        for _ in range(NUM_ITERATIONS):
           mnist_classifier.train(
               input_fn=train_input_fn,
               steps=40)

           eval_results = mnist_classifier.evaluate(input_fn=eval_input_fn)
           print(eval_results)

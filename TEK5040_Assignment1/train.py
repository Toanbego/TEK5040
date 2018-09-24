from __future__ import print_function

import glob
import os
import time
import argparse

import tensorflow as tf


def generator_for_filenames(*filenames):
    """
    Wrapping a list of filenames as a generator function
    """
    def generator():
        for f in zip(*filenames):
            yield f
    return generator


def preprocess(image, segmentation):
    """
    A preprocess function the is run after images are read. Here you can do augmentation and other
    processesing on the images.
    """
    # Set images size to a constant

    image = tf.image.resize_images(image, [256, 256])
    segmentation = tf.image.resize_images(segmentation, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

    image = tf.to_float(image) / 255
    segmentation = tf.to_int64(segmentation)

    # Do some processing

    # Add random colours
    image = tf.image.random_hue(image, 0.25)
    #
    # # Random flips
    # image = tf.image.random_flip_up_down(image)


    return image, segmentation


def read_image_and_segmentation(img_f, seg_f):
    """
    Read images from file using tensorflow and convert the segmentation to appropriate formate.
    :param img_f: filename for image
    :param seg_f: filename for segmentation
    :return: Image and segmentation tensors
    """
    img_reader = tf.read_file(img_f)
    seg_reader = tf.read_file(seg_f)
    img = tf.image.decode_png(img_reader, channels=3)
    seg = tf.image.decode_png(seg_reader)[:, :, 2:]
    seg = tf.where(seg > 0, tf.ones_like(seg), tf.zeros_like(seg))
    return img, seg


def kitti_generator_from_filenames(image_names, segmentation_names, preprocess=preprocess, batch_size=8):
    """
    Convert a list of filenames to tensorflow images.
    :param image_names: image filenames
    :param segmentation_names: segmentation filenames
    :param preprocess: A function that is run after the images are read, the takes image and
    segmentation as input
    :param batch_size: The batch size returned from the function
    :return: Tensors with images and corresponding segmentations
    """
    dataset = tf.data.Dataset.from_generator(
        generator_for_filenames(image_names, segmentation_names),
        output_types=(tf.string, tf.string),
        output_shapes=(None, None)
    )

    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.map(read_image_and_segmentation)
    dataset = dataset.map(preprocess)
    dataset = dataset.repeat()
    dataset = dataset.batch(batch_size)

    return dataset.repeat().make_one_shot_iterator().get_next()


def kitti_image_filenames(dataset_folder, training=True):
    sub_dataset = 'training' if training else 'testing'
    segmentation_names = glob.glob(os.path.join(dataset_folder, sub_dataset, 'gt_image_2', '*road*.png'),
                                   recursive=True)
    image_names = [f.replace('gt_image_2', 'image_2').replace('_road_', '_') for f in segmentation_names]
    return image_names, segmentation_names


def improved_model(args, img, seg, training):
    """
    Improved model uses a structure that downsamples the image, and then
    upsamples again. Following an encoder/decoder structure.

    Uses cross_entropy as loss functions since it handles pixels on an
    individual level.
    :param img:
    :param seg:
    :return:
    """

    if args.regularization == True:

        regularizer = tf.contrib.layers.l2_regularizer(0.001)
        # Downsample
        x = tf.layers.conv2d(img, 16, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d(x, 16, kernel_size=5, strides=(2, 2), padding='same',
                           activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d(x, 32, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d(x, 64, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d(x, 128, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu, kernel_regularizer=regularizer)

        # Upsample
        x = tf.layers.conv2d_transpose(x, 64, kernel_size=3, strides=(2, 2), padding='same',
                                       activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d_transpose(x, 32, kernel_size=3, strides=(2, 2), padding='same',
                                       activation=tf.nn.relu, kernel_regularizer=regularizer)
        x = tf.layers.conv2d_transpose(x, 16, kernel_size=1, strides=(1, 1), padding='same',
                                       activation=tf.nn.relu, kernel_regularizer=regularizer)

    elif args.regularization != True:

        # Downsample

        x = tf.layers.conv2d(img, 32, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 64, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 128, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 256, kernel_size=5, strides=(2, 2), padding='same',
                             activation=tf.nn.relu)

        # Upsample
        x = tf.layers.conv2d_transpose(x, 128, kernel_size=5, strides=(2, 2), padding='same',
                                       activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 64, kernel_size=5, strides=(2, 2), padding='same',
                                       activation=tf.nn.relu)
        x = tf.layers.conv2d_transpose(x, 32, kernel_size=5, strides=(2, 2), padding='same',
                                       activation=tf.nn.relu)

    # Add some dropout to prevent overfitting
    x = tf.layers.dropout(x, rate=0.25, training=training)

    # Uses sigmoid to create values between 0 and 1 for more distinct segmentation.
    x = tf.layers.conv2d(x, 1, kernel_size=1, padding='same',
                         activation=tf.nn.sigmoid)

    # Resize image
    x = tf.image.resize_images(x, [256, 256])

    # Cross_entropy with sigmoid (maximumllikelihood for bernoulliooierio random variabler)
    cross = tf.to_float(seg) * tf.log(1e-3 + x) + (1 - tf.to_float(seg)) * tf.log((1-x) + 1e-3)
    loss = tf.reduce_mean(-cross + tf.losses.get_regularization_loss())

    return x, loss


def model(img, seg):
    x = tf.layers.conv2d(img, 32, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 64, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)

    x = tf.layers.conv2d(x, 1, kernel_size=1, padding='same')
    # x = tf.layers.flatten(x)

    x = tf.image.resize_images(x, [256, 256])

    loss = tf.losses.mean_squared_error(seg, x)
    loss = tf.reduce_mean(loss)
    return x, loss


def create_tensorboard_summaries(img, img_val, seg_val, logits_val, logits, seg, loss, loss_val, step):
    """
    Define metrics for tensorboard
    :param img:
    :param img_val:
    :param seg_val:
    :param logits_val:
    :param logits:
    :param seg:
    :param loss:
    :param loss_val:
    :param step:
    :return:
    """
    # Calculating metrics to calculate F1 score
    accuracy, acc = tf.metrics.accuracy(seg, logits)
    precision, prec = tf.metrics.precision(seg, logits)
    recall, rec = tf.metrics.recall(seg, logits)

    accuracy_, acc_ = tf.metrics.accuracy(seg_val, logits_val)
    precision_, prec_ = tf.metrics.precision(seg_val, logits_val)
    recall_, rec_ = tf.metrics.recall(seg_val, logits_val)

    F1 = 2 * prec * rec / (prec + rec)
    F1_ = 2 * prec * rec / (prec + rec)

    # Tensorboard for metrics
    # Training
    tf.summary.scalar('loss', loss, family='training')
    tf.summary.scalar('accuracy_train', acc, family='training')
    tf.summary.scalar('precision', prec, family='training')
    tf.summary.scalar('recall', rec, family='training')
    tf.summary.scalar('F_measure', F1, family='training')

    # Validation
    tf.summary.scalar('loss_val', loss_val, family='validation')
    tf.summary.scalar('accuracy_val', acc_, family='validation')
    tf.summary.scalar('precision_val', prec_, family='validation')
    tf.summary.scalar('recall_val', rec_, family='validation')
    tf.summary.scalar('F_measure_val', F1_, family='validation')
    tf.summary.scalar('step', step)


    # Tensorboard for images
    zeros = tf.zeros(tf.shape(logits))

    # Traning images
    superimposed = img + tf.concat(axis=3, values=(zeros, logits, zeros))
    tf.summary.image('Image - Training', img, max_outputs=1, family='training')
    tf.summary.image('Image - Training_pred', logits, max_outputs=1, family='training')
    tf.summary.image('Image - Training_label', tf.cast(seg, dtype=tf.float32), max_outputs=1, family='training')
    tf.summary.image('Image - Training_overlay', superimposed, max_outputs=1, family='training')

    # Validation image
    superimposed_ = img_val + tf.concat(axis=3, values=(zeros, logits_val, zeros))
    tf.summary.image('Image - Validation', img_val, max_outputs=1, family='validation')
    tf.summary.image('Image - Validation_pred', logits_val, max_outputs=1, family='validation')
    tf.summary.image('Image - Validation_label', tf.cast(seg_val, dtype=tf.float32), max_outputs=1, family='validation')
    tf.summary.image('Image - Valdiation_overlay', superimposed_, max_outputs=1, family='validation')


def parse_arguments():
    parser = argparse.ArgumentParser()
    # Set up arguments

    parser.add_argument('--model', '-m', type=str, default='improved',
                        help='Which model to use')
    parser.add_argument('--regularization', '-l', type=bool, default=False,
                        help='Whether or not to use l2 regularization')

    args = parser.parse_args()
    return args


def main(_):

    # Parse arguments
    args = parse_arguments()

    # Getting filenames from the kitti dataset
    image_names, segmentation_names = kitti_image_filenames('data_road')

    # Get image tensors from the filenames
    img, seg = kitti_generator_from_filenames(
        image_names[:-3],
        segmentation_names[:-3],
        batch_size=8)
    # Get the validation tensors
    img_val, seg_val = kitti_generator_from_filenames(
        image_names[-3:],
        segmentation_names[-3:],
        batch_size=8)

    if args.model == 'improved':
        # Create the model
        with tf.variable_scope('model'):
            logits, loss = improved_model(args, img, seg, training=True)

        # Reuse the same model for validation
        with tf.variable_scope('model', reuse=True):
            logits_val, loss_val = improved_model(args, img_val, seg_val, training=False)

    elif args.model == 'standard':
        # Create the model
        with tf.variable_scope('model'):
            logits, loss = model(img, seg)

        # Reuse the same model for validation
        with tf.variable_scope('model', reuse=True):
            logits_val, loss_val = model(img_val, seg_val)

    # Keep track of number of steps
    step = tf.train.get_or_create_global_step()

    # Create an optimizer
    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
        train_op = tf.train.AdamOptimizer(0.001).minimize(
            loss,
            global_step=step)

    # Sets up the summary for the various metrics
    create_tensorboard_summaries(img, img_val, seg_val,
                                 logits_val, logits,
                                 seg, loss,
                                 loss_val, step)


    # Creates summary file
    saver_hook = tf.train.CheckpointSaverHook('logs', save_secs=10)
    summary_saver_hook = tf.train.SummarySaverHook(
        summary_op=tf.summary.merge_all(),
        output_dir='logs',
        save_secs=10
    )

    #Run training
    with tf.train.SingularMonitoredSession(
            hooks=[saver_hook, summary_saver_hook],
            checkpoint_dir='logs') as sess:
        # TF placeholders

        while not sess.should_stop():
            _, loss_, step_ = sess.run([train_op, loss, step])

            print(step_, 'loss', loss_)
            if step_ % 10 == 0:
                loss_ = sess.run([loss_val])
                print('\t\t\tval_loss', loss_)

if __name__ == '__main__':
    main(None)

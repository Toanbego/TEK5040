from __future__ import print_function

import glob
import os
import time

import tensorflow as tf
import tensorboard

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

def improved_model(img, seg):
    """
    Improved model uses a structure that downsamples the image, and then
    upsamples again. Following an encoder/decoder structure.

    Uses cross_entropy as loss functions since it handles pixels on an
    individual level.
    :param img:
    :param seg:
    :return:
    """
    # First layer
    x = tf.layers.conv2d(img, 32, kernel_size=3, strides=(2, 2), padding='same',
                         activation=tf.nn.relu)

    # Down sample
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=(2, 2), padding='same',
                         activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 64, kernel_size=3, strides=(2, 2), padding='same',
                         activation=tf.nn.relu)

    # Up sample

    x = tf.layers.conv2d_transpose(x, 32, kernel_size=3, strides=(2, 2), padding='same',
                                   activation=tf.nn.relu)
    x = tf.layers.conv2d_transpose(x, 24, kernel_size=3, strides=(2, 2), padding='same',
                                   activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 1, kernel_size=1, padding='same',
                         activation=tf.nn.sigmoid)

    x = tf.image.resize_images(x, [256, 256])


    # Cross_entropy with sigmoid (maximumllikelihood for bernoulliooierio random variabler)
    cross = tf.to_float(seg) * tf.log(1e-3 + x) + (1 - tf.to_float(seg)) * tf.log((1-x) + 1e-3)

    loss = tf.reduce_mean(-cross)

    return x, loss


def model(img, seg):
    """

    :param img:
    :param seg:
    :return:
    """
    x = tf.layers.conv2d(img, 128, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 64, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 32, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 64, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 128, kernel_size=5, strides=(2, 2), padding='same', activation=tf.nn.relu)
    x = tf.layers.conv2d(x, 1, kernel_size=1, padding='same')  # Fully connected?
    # x = tf.layers.flatten(x)

    x = tf.image.resize_images(x, [512, 512])


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

    accuracy, acc = tf.metrics.accuracy(seg_val, logits_val)
    precision, prec = tf.metrics.precision(seg_val, logits_val)
    recall, rec = tf.metrics.recall(seg_val, logits_val)

    F1 = 2 * prec * rec / (prec + rec)

    # Tensorboard for metrics
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('loss_val', loss_val)
    tf.summary.scalar('step', step)
    tf.summary.scalar('accuracy', acc)
    tf.summary.scalar('precision', prec)
    tf.summary.scalar('recall', rec)
    tf.summary.scalar('F_measure', F1)

    # Tensorboard for images
    # tf.summary.image('Image Validation Label', seg_val, max_outputs=1)
    tf.summary.image('Image - Validation', img_val, max_outputs=1)
    tf.summary.image('Image - seg', tf.cast(seg, dtype=tf.float32), max_outputs=1)
    tf.summary.image('Image - Validation_pred', logits_val, max_outputs=1)
    tf.summary.image('Image - Training', img, max_outputs=1)
    tf.summary.image('Image - Training_pred', logits, max_outputs=1)


def main(_):
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

    #Create the model
    with tf.variable_scope('model'):
        logits, loss = improved_model(img, seg)

    # Reuse the same model for validation
    with tf.variable_scope('model', reuse=True):
        logits_val, loss_val = improved_model(img_val, seg_val)

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

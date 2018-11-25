import os

import tensorflow as tf


def lpips(input0, input1, model='net-lin', net='alex', version=0.1):
    """
    Learned Perceptual Image Patch Similarity (LPIPS) metric.

    Args:
        input0: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].
        input1: An image tensor of shape `[..., height, width, channels]`,
            with values in [0, 1].

    Returns:
        The Learned Perceptual Image Patch Similarity (LPIPS) distance.

    Refenrece:
        Richard Zhang, Phillip Isola, Alexei A. Efros, Eli Shechtman, Oliver Wang.
        The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.
        In CVPR, 2018.
    """
    # flatten the leading dimensions
    batch_shape = tf.shape(input0)[:-3]
    input0 = tf.reshape(input0, tf.concat([[-1], tf.shape(input0)[-3:]], axis=0))
    input1 = tf.reshape(input1, tf.concat([[-1], tf.shape(input1)[-3:]], axis=0))
    # NCHW to NHWC
    input0 = tf.transpose(input0, [0, 3, 1, 2])
    input1 = tf.transpose(input1, [0, 3, 1, 2])
    # normalize to [-1, 1]
    input0 = input0 * 2.0 - 1.0
    input1 = input1 * 2.0 - 1.0

    input0_name, input1_name = '0:0', '1:0'
    pb_fname = os.path.join(os.path.dirname(__file__), 'models/v%s/%s_%s.pb' % (version, model, net))
    with open(pb_fname, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def,
                                input_map={input0_name: input0, input1_name: input1})
        distance, = tf.get_default_graph().get_operations()[-1].outputs

    if distance.shape.ndims == 4:
        distance = tf.squeeze(distance, axis=[-3, -2, -1])
    # reshape the leading dimensions
    distance = tf.reshape(distance, batch_shape)
    return distance

import argparse

import cv2
import numpy as np
import tensorflow as tf

import lpips_tf


def load_image(fname):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['net-lin', 'net'], default='net-lin', help='net-lin or net')
    parser.add_argument('--net', choices=['squeeze', 'alex', 'vgg'], default='alex', help='squeeze, alex, or vgg')
    parser.add_argument('--version', type=str, default='0.1')
    args = parser.parse_args()

    ex_ref = load_image('./PerceptualSimilarity/imgs/ex_ref.png')
    ex_p0 = load_image('./PerceptualSimilarity/imgs/ex_p0.png')
    ex_p1 = load_image('./PerceptualSimilarity/imgs/ex_p1.png')

    session = tf.Session()

    image0_ph = tf.placeholder(tf.float32)
    image1_ph = tf.placeholder(tf.float32)
    lpips_fn = session.make_callable(
        lpips_tf.lpips(image0_ph, image1_ph, model=args.model, net=args.net, version=args.version),
        [image0_ph, image1_ph])

    ex_d0 = lpips_fn(ex_ref, ex_p0)
    ex_d1 = lpips_fn(ex_ref, ex_p1)

    print('Distances: (%.3f, %.3f)' % (ex_d0, ex_d1))


if __name__ == '__main__':
    main()

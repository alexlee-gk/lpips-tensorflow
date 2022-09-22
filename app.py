import streamlit as st
import cv2
from PIL import Image
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import lpips_tf

def load_image(fname):
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image.astype(np.float32) / 255.0

net = st.radio("Pick a model", ['vgg', 'alex', 'squeeze'])
uploaded_file = st.file_uploader("Choose a file", key="0")
uploaded_file1 = st.file_uploader("Choose a file", key="1")
if uploaded_file is not None and uploaded_file1 is not None:
    ex_ref = load_image(str(uploaded_file.name))
    image = Image.open(uploaded_file)
    #col1, col2= st.columns([1, 45])
    st.image(uploaded_file, caption='Reference Image', use_column_width=False)
    ex_p0 = load_image(str(uploaded_file1.name))
    image1 = Image.open(uploaded_file1)
    st.image(uploaded_file1, caption='Input Image', use_column_width=False)

if st.button('Get the distance'):
    if uploaded_file is not None and uploaded_file1 is not None:
        model = 'net-lin'
        version = '0.1'
        session = tf.Session()
        image0_ph = tf.placeholder(tf.float32)
        image1_ph = tf.placeholder(tf.float32)
        lpips_fn = session.make_callable(
                lpips_tf.lpips(image0_ph, image1_ph, model=model, net=net, version=version),
                [image0_ph, image1_ph])

        ex_d0 = lpips_fn(ex_ref, ex_p0)
        output = 'Distances: (%.3f)' % ex_d0
        st.write(output)

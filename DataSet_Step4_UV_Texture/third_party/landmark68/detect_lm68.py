import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf
from shutil import move

BBRegressorParam = loadmat('third_party/landmark68/BBRegressorParam_r.mat')
mean_face = np.loadtxt('third_party/landmark68/test_mean_face.txt')
mean_face = mean_face.reshape([68, 2])


# utils for landmark detection
def img_padding(img, box):
    success = True
    bbox = box.copy()
    res = np.zeros([2 * img.shape[0], 2 * img.shape[1], 3])
    res[img.shape[0] // 2:img.shape[0] + img.shape[0] // 2, img.shape[1] // 2:img.shape[1] + img.shape[1] // 2] = img

    bbox[0] = bbox[0] + img.shape[1] // 2
    bbox[1] = bbox[1] + img.shape[0] // 2
    if bbox[0] < 0 or bbox[1] < 0:
        success = False
    return res, bbox, success


# utils for landmark detection
def crop(img, bbox):
    padded_img, padded_bbox, flag = img_padding(img, bbox)
    if flag:
        crop_img = padded_img[padded_bbox[1]:padded_bbox[1] + padded_bbox[3],
                              padded_bbox[0]:padded_bbox[0] + padded_bbox[2]]
        crop_img = cv2.resize(crop_img.astype(np.uint8), (224, 224), interpolation=cv2.INTER_CUBIC)
        scale = 224 / padded_bbox[3]
        return crop_img, scale
    else:
        return padded_img, 0


# bounding box for 68 landmark detection
def BBRegression(points, params):

    w1 = params['W1']
    b1 = params['B1']
    w2 = params['W2']
    b2 = params['B2']
    data = points.copy()
    data = data.reshape([5, 2])
    data_mean = np.mean(data, axis=0)
    x_mean = data_mean[0]
    y_mean = data_mean[1]
    data[:, 0] = data[:, 0] - x_mean
    data[:, 1] = data[:, 1] - y_mean

    rms = np.sqrt(np.sum(data**2) / 5)
    data = data / rms
    data = data.reshape([1, 10])
    data = np.transpose(data)
    inputs = np.matmul(w1, data) + b1
    inputs = 2 / (1 + np.exp(-2 * inputs)) - 1
    inputs = np.matmul(w2, inputs) + b2
    inputs = np.transpose(inputs)
    x = inputs[:, 0] * rms + x_mean
    y = inputs[:, 1] * rms + y_mean
    w = 224 / inputs[:, 2] * rms
    rects = [x, y, w, w]
    return np.array(rects).reshape([4])


# utils for landmark detection
def align_for_lm(img, five_points):
    five_points = np.array(five_points).reshape([1, 10])
    bbox = BBRegression(five_points, BBRegressorParam)
    assert (bbox[2] != 0)
    bbox = np.round(bbox).astype(np.int32)
    crop_img, scale = crop(img, bbox)
    return crop_img, scale, bbox


def detect_68p(img, five_points, sess, input_op, output_op):

    img = np.array(img)  ## transfer to numpy, RGB
    input_img, scale, bbox = align_for_lm(img, five_points)  # align for 68 landmark detection

    # detect landmarks
    input_img = np.reshape(input_img, [1, 224, 224, 3]).astype(np.float32)
    landmark = sess.run(output_op, feed_dict={input_op: input_img})

    # transform back to original image coordinate
    landmark = landmark.reshape([68, 2]) + mean_face
    landmark[:, 1] = 223 - landmark[:, 1]
    landmark = landmark / scale
    landmark[:, 0] = landmark[:, 0] + bbox[0]
    landmark[:, 1] = landmark[:, 1] + bbox[1]
    landmark[:, 1] = img.shape[0] - 1 - landmark[:, 1]

    return landmark


# create tensorflow graph for landmark detector
def load_lm_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='net')
        img_224 = graph.get_tensor_by_name('net/input_imgs:0')
        output_lm = graph.get_tensor_by_name('net/lm:0')
        # lm_sess = tf.Session(graph=graph)
        lm_sess = tf.InteractiveSession(graph=graph)

    return lm_sess, img_224, output_lm

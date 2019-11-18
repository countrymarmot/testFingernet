from __future__ import absolute_import
from __future__ import print_function

import os
import cv2
import numpy as np
import random
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

from keras.optimizers import SGD, Adam, RMSprop
from keras.models import Model, Sequential
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout, Lambda
from keras import backend as K

############################# prepare training data ######################################


def load_training_img(directory):
    fingerprints = []
    labels = []

    if not os.path.exists(directory):
        print("directory " + directory + " doesn't exists.")

    for subdir in os.listdir(directory):
        sub_directory = directory + subdir + "/"
        if not os.path.isdir(sub_directory):
            print("error: Not dir, %s" % sub_directory)
            continue

        filenum = 0
        for f in os.listdir(sub_directory):
            file = sub_directory + f
            if not os.path.isfile(file):
                continue
            filename, ext = os.path.splitext(f)
            if ext.upper() != ".BMP":
                continue
            fps = cv2.imread(file)
            if fps is not None:
                fingerprints.append(fps[:, :, 0])  # only need 1 channel for gray image
                # fingerprints.append(fps)
                labels.append(subdir)
                filenum += 1
        print("load %d samples from %s" % (filenum, subdir))

    return fingerprints, labels


def create_pairs(x, digit_indices, num_classes):
    '''Positive and negative pair creation.
    Alternates between positive and negative pairs.
    '''
    pairs = []
    labels = []
    n = min([len(digit_indices[d]) for d in range(num_classes)]) - 1
    for d in range(num_classes):
        for i in range(n):
            z1, z2 = digit_indices[d][i], digit_indices[d][i + 1]
            pairs += [[x[z1], x[z2]]]
            inc = random.randrange(1, num_classes)
            dn = (d + inc) % num_classes
            z1, z2 = digit_indices[d][i], digit_indices[dn][i]
            pairs += [[x[z1], x[z2]]]
            labels += [1, 0]
    return np.array(pairs), np.array(labels)


def prepare_training_data():

    # load raw images
    X_full, y_full = load_training_img(r"E:/deeplearning/fingerprints/denali/Overall_DB/Overall/")
    #height, width, channel = X_full[0].shape
    height, width = X_full[0].shape
    channel = 1  # only gray image
    class_num = len(Counter(y_full))
    im_input_shape = (height, width, channel)

    # transfer the label
    le = preprocessing.LabelEncoder()
    le.fit(y_full)
    y_full_labelled = le.transform(y_full)

    # split the training and test set
    X_tr, X_vl, y_tr, y_vl = train_test_split(X_full, y_full_labelled, test_size=0.3, random_state=3)

    # create training+test positive and negative pairs
    digit_indices = [np.where(y_tr == i)[0] for i in range(class_num)]
    tr_pairs, tr_y = create_pairs(X_tr, digit_indices, class_num)

    digit_indices = [np.where(y_vl == i)[0] for i in range(class_num)]
    te_pairs, te_y = create_pairs(X_vl, digit_indices, class_num)

    return (tr_pairs, tr_y), (te_pairs, te_y), im_input_shape


############################# create network ######################################


def euclidean_distance(vects):
    x, y = vects
    sum_square = K.sum(K.square(x - y), axis=1, keepdims=True)
    return K.sqrt(K.maximum(sum_square, K.epsilon()))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)


def create_share_network(input_shape):
    input = Input(shape=input_shape)
    x = Conv2D(kernel_size=(3, 3), filters=20, activation='relu')(input)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(kernel_size=(3, 3), filters=20, activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(kernel_size=(3, 3), filters=20, activation='relu')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu', name="feature")(x)
    return Model(input, x)


def create_network(im_input_shape):
    a = Input(im_input_shape)
    b = Input(im_input_shape)

    share_network = create_share_network(im_input_shape)
    model_a = share_network(a)
    model_b = share_network(b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([model_a, model_b])
    m_model = Model([a, b], distance)
    m_model.summary()

    return m_model

    #model = Model([input_a, input_b], distance)
    #final_layer = Conv2D(kernel_size=(1, 1), filters=25, activation='relu')(normalized_layer)
    #final_layer = Conv2D(kernel_size=(3, 3), filters=25, activation=None)(final_layer)
    #final_layer = MaxPooling2D((2, 2))(final_layer)
    #final_layer = Dense(500)(final_layer)
    #final_layer = Dense(1, activation="sigmoid")(final_layer)
    #x_corr_mod = Model(inputs=[a, b], outputs=final_layer)
    #try:
    #    x_corr_mod.summary()
    #except:
    #    pass
    #print(x_corr_mod.output._keras_shape)
    #return x_corr_mod


def compute_accuracy(y_true, y_pred):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    pred = y_pred.ravel() < 0.5
    return np.mean(pred == y_true)

################################################################################

if __name__ == "__main__":

    (tr_pairs, tr_y), (te_pairs, te_y), im_input_shape = prepare_training_data()
    #quantity, height, width = tr_pairs[:, 0].shape
    #im_input_shape = (height, width, 1)

    print(im_input_shape)

    # need expand dims for keras
    X_train_a = np.expand_dims(tr_pairs[:, 0], -1)
    X_train_b = np.expand_dims(tr_pairs[:, 1], -1)

    X_valid_a = np.expand_dims(te_pairs[:, 0], -1)
    X_valid_b = np.expand_dims(te_pairs[:, 1], -1)

    # prepare model
    m_model = create_network(im_input_shape)
    m_model.compile(loss=contrastive_loss, optimizer=Adam(lr=0.0001, decay=1e-6))
    m_model.fit([X_train_a, X_train_b], tr_y, batch_size=64,
                shuffle=True,
                verbose=2,
                epochs=10,
                validation_data=([X_valid_a, X_valid_b], te_y))

# compute final accuracy on training and test sets
y_pred = m_model.predict([X_train_a, X_train_b])
tr_acc = compute_accuracy(tr_y, y_pred)
y_pred = m_model.predict([X_valid_a, X_valid_b])
te_acc = compute_accuracy(te_y, y_pred)

print('* Accuracy on training set: %0.2f%%' % (100 * tr_acc))
print('* Accuracy on test set: %0.2f%%' % (100 * te_acc))

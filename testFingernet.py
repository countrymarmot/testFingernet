import model
import os
import cv2
import numpy as np
import random
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from keras.optimizers import SGD, Adam

# prepare training data


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

    return (tr_pairs, tr_y), (te_pairs, te_y)



if __name__ == "__main__":

    (tr_pairs, tr_y), (te_pairs, te_y) = prepare_training_data()
    print(tr_pairs[:, 0].shape)

    quantity, height, width = tr_pairs[:, 0].shape
    im_input_shape = (height, width, 1)
    print(im_input_shape)

    # prepare model
    model.normalized_x_corr_model(im_input_shape)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=0.0001, decay=1e-6))
    history = model.fit([tr_pairs[:, 0], tr_pairs[:, 1]], tr_y, batch_size=64, shuffle=True, verbose=2, epochs=10)


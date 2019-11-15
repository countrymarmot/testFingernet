import model
import os
import cv2


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


# prepare model
model.normalized_x_corr_model()

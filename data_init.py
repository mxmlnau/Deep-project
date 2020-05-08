import os
import numpy as np
import nibabel as nib
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array
from skimage.transform import resize
from matplotlib import pyplot as plt
import tensorflow as tf
from unet import model


def concatenate_data(x_path, y_path, l_path):
    print("Concatenating...")
    for i in range(9):
        curr_x = nib.load(x_path + '/' + str(i+1) + '.nii').get_fdata()
        curr_y = nib.load(y_path + '/' + str(i+1) + '.nii').get_fdata()
        curr_l = nib.load(l_path + '/' + str(i+1) + '.nii').get_fdata()

        if i == 0:
            x = curr_x
            y = curr_y
            l = curr_l
        else:
            x = np.concatenate((x, curr_x), axis=2)
            y = np.concatenate((y, curr_y), axis=2)
            l = np.concatenate((l, curr_l), axis=2)

    return x.T, y.T, l.T


def resize_and_flip(x, y, l):
    print("Resizing..")
    num_of_imgs = np.size(x, axis=2)

    resized_x = np.zeros((num_of_imgs, 512, 512, 1))
    resized_y = np.zeros((num_of_imgs, 512, 512, 1))
    resized_l = np.zeros((num_of_imgs, 512, 512, 1))

    all_data = [[x, resized_x], [y, resized_y], [l, resized_l]]

    for data in all_data:
        for i in range(num_of_imgs):
            resized_img = resize(data[0][i], (512, 512), preserve_range=True)

            #flipped_resized_img = np.fliplr(np.rot90(resized_img, k=3))

            data[1][i] = img_to_array(resized_img)

    return resized_x, resized_y, resized_l


def augment_init(x, y):
    data_gen_args = dict(featurewise_center=True,
                         featurewise_std_normalization=True,
                         rotation_range=90,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)

    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)

    seed = 1
    image_datagen.fit(x, augment=True, seed=seed)
    mask_datagen.fit(y, augment=True, seed=seed)

    image_generator = image_datagen.flow(
        x,
        batch_size=1,
        seed=seed)

    mask_generator = mask_datagen.flow(
        y,
        batch_size=1,
        seed=seed)

    train_generator = zip(image_generator, mask_generator)

    return train_generator


def main():
    x_path = 'data/2/rp_im'
    y_path = 'data/2/rp_msk'
    l_path = 'data/2/rp_lung_msk'

    x, y, l = concatenate_data(x_path, y_path, l_path)

    x, y, l = resize_and_flip(x, y, l)

    train_gen = augment_init(x, y)

    model.fit(train_gen, epochs=2)


if __name__ == "__main__":
    main()

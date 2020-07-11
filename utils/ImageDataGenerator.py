"""
The implementation of Data Generator based on keras.


# Reference
    https://github.com/luyanger1799/amazing-semantic-segmentation
    keras 2.0.8: keras/preprocessing/imgage.py
    keras 2.1.6: keras/preprocessing/imgage.py  reimplement Iterator and Sequence
"""
# from tensorflow.python.keras.preprocessing.image import Iterator
# from keras.preprocessing.image import Iterator
from keras.applications import imagenet_utils
from keras.preprocessing import image as keras_image
#from tf.keras.preprocessing import image
from abc import ABCMeta, abstractmethod
import tensorflow as tf
import threading
from PIL import Image
import cv2
import numpy as np

keras_utils = tf.keras.utils


def resize_image(image, label, target_size=None):
    if target_size is not None:
        image = cv2.resize(image, dsize=target_size[::-1])
        label = cv2.resize(
            label, dsize=target_size[::-1], interpolation=cv2.INTER_NEAREST)
    return image, label


def load_image(name):
    img = Image.open(name)
    return np.array(img)


def random_crop(image, label, crop_size):
    h, w = image.shape[0:2]
    crop_h, crop_w = crop_size

    if h < crop_h or w < crop_w:
        image = cv2.resize(image, (max(w, crop_w), max(h, crop_h)))
        label = cv2.resize(label, (max(w, crop_w), max(
            h, crop_h)), interpolation=cv2.INTER_NEAREST)

    h, w = image.shape[0:2]

    if h - crop_h == 0:
        h_beg = 0
    else:
        h_beg = np.random.randint(h - crop_h)

    if w - crop_w == 0:
        w_beg = 0
    else:
        w_beg = np.random.randint(w - crop_w)

    cropped_image = image[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]
    cropped_label = label[h_beg:h_beg + crop_h, w_beg:w_beg + crop_w]

    return cropped_image, cropped_label


def random_zoom(image, zoom_range):


    if np.isscalar(zoom_range):
        zx, zy = np.random.uniform(1 - zoom_range, 1 + zoom_range, 2)
    elif len(zoom_range) == 2:
        zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
    else:
        raise ValueError('`zoom_range` should be a float or '
                         'a tuple or list of two floats. '
                         'Received: %s' % (zoom_range,))

    image = apply_affine_transform(
        image, zx=zx, zy=zy, fill_mode='nearest')


    return image


def random_brightness(image, label, brightness_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if brightness_range is not None:
        if isinstance(brightness_range, (tuple, list)) and len(brightness_range) == 2:
            brightness = np.random.uniform(
                brightness_range[0], brightness_range[1])
        else:
            raise ValueError('`brightness_range` should be '
                             'a tuple or list of two floats. '
                             'Received: %s' % (brightness_range,))
        image = keras_image.apply_brightness_shift(image, brightness)
    return image, label


def random_horizontal_flip(image, label, h_flip):
    if h_flip:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)
    return image, label


def random_vertical_flip(image, label, v_flip):
    if v_flip:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)
    return image, label


def random_rotation(image, label, rotation_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if rotation_range > 0.:
        theta = np.random.uniform(-rotation_range, rotation_range)
        # rotate it!
        image = keras_image.apply_affine_transform(
            image, theta=theta, fill_mode='nearest')
        label = keras_image.apply_affine_transform(
            label, theta=theta, fill_mode='nearest')
    return image, label


def random_channel_shift(image, label, channel_shift_range):
    if np.ndim(label) == 2:
        label = np.expand_dims(label, axis=-1)
    assert np.ndim(label) == 3

    if channel_shift_range > 0:
        channel_shift_intensity = np.random.uniform(
            -channel_shift_range, channel_shift_range)
        image = keras_image.apply_channel_shift(
            image, channel_shift_intensity, channel_axis=2)
    return image, label


def one_hot(label, num_classes):
    if np.ndim(label) == 3:
        label = np.squeeze(label, axis=-1)
    assert np.ndim(label) == 2

    heat_map = np.ones(shape=label.shape[0:2] + (num_classes,))
    for i in range(num_classes):
        heat_map[:, :, i] = np.equal(label, i).astype('float32')
    return heat_map


def decode_one_hot(one_hot_map):
    return np.argmax(one_hot_map, axis=-1)


class Sequence(object):
    """Base object for fitting to a sequence of data, such as a dataset.

    Every `Sequence` must implements the `__getitem__` and the `__len__` methods.
    If you want to modify your dataset between epochs you may implement `on_epoch_end`.
    The method `__getitem__` should return a complete batch.

    # Notes

    `Sequence` are a safer way to do multiprocessing. This structure guarantees that the network will only train once
     on each sample per epoch which is not the case with generators.

    """

    @abstractmethod
    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        raise NotImplementedError

    @abstractmethod
    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        raise NotImplementedError

    def on_epoch_end(self):
        """Method called at the end of every epoch.
        """
        pass

    def __iter__(self):
        """Create an infinite generator that iterate over the Sequence."""
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item


class Iterator(Sequence):
    """Base class for image data iterators.

    Every `Iterator` must implement the `_get_batches_of_transformed_samples`
    method.

    # Arguments
        n: Integer, total number of samples in the dataset to loop over.
        batch_size: Integer, size of a batch.
        shuffle: Boolean, whether to shuffle the data between epochs.
        seed: Random seeding for data shuffling.
    """

    def __init__(self, n, batch_size, shuffle, seed):
        self.n = n
        self.batch_size = batch_size
        self.seed = seed
        self.shuffle = shuffle
        self.batch_index = 0
        self.total_batches_seen = 0
        self.lock = threading.Lock()
        self.index_array = None
        self.index_generator = self._flow_index()

    def _set_index_array(self):
        self.index_array = np.arange(self.n)
        if self.shuffle:
            self.index_array = np.random.permutation(self.n)

    def __getitem__(self, idx):
        if idx >= len(self):
            raise ValueError('Asked to retrieve element {idx}, '
                             'but the Sequence '
                             'has length {length}'.format(idx=idx,
                                                          length=len(self)))
        if self.seed is not None:
            np.random.seed(self.seed + self.total_batches_seen)
        self.total_batches_seen += 1
        if self.index_array is None:
            self._set_index_array()
        index_array = self.index_array[self.batch_size * idx:
                                       self.batch_size * (idx + 1)]
        return self._get_batches_of_transformed_samples(index_array)

    def __len__(self):
        return (self.n + self.batch_size - 1) // self.batch_size  # round up

    def on_epoch_end(self):
        self._set_index_array()

    def reset(self):
        self.batch_index = 0

    def _flow_index(self):
        # Ensure self.batch_index is 0.
        self.reset()
        while 1:
            if self.seed is not None:
                np.random.seed(self.seed + self.total_batches_seen)
            if self.batch_index == 0:
                self._set_index_array()

            current_index = (self.batch_index * self.batch_size) % self.n
            if self.n > current_index + self.batch_size:
                self.batch_index += 1
            else:
                self.batch_index = 0
            self.total_batches_seen += 1
            yield self.index_array[current_index:
                                   current_index + self.batch_size]

    def __iter__(self):
        # Needed if we want to do something like:
        # for x, y in data_gen.flow(...):
        return self

    def __next__(self, *args, **kwargs):
        return self.next(*args, **kwargs)

    def _get_batches_of_transformed_samples(self, index_array):
        """Gets a batch of transformed samples.

        # Arguments
            index_array: array of sample indices to include in batch.

        # Returns
            A batch of transformed samples.
        """
        raise NotImplementedError


class DataIterator(Iterator):
    def __init__(self,
                 image_data_generator,
                 images_list,
                 labels_list,
                 num_classes,
                 batch_size,
                 target_size,
                 pad_size,
                 shuffle=True,
                 seed=None,
                 data_aug_rate=0.):
        num_images = len(images_list)

        self.image_data_generator = image_data_generator
        self.images_list = images_list
        self.labels_list = labels_list
        self.num_classes = num_classes
        self.target_size = target_size
        self.pad_size = pad_size
        self.data_aug_rate = data_aug_rate

        super(DataIterator, self).__init__(
            num_images, batch_size, shuffle, seed)

    def _get_batches_of_transformed_samples(self, index_array):
        batch_x = np.zeros(shape=(len(index_array),) + self.target_size + (3,))
        batch_y = np.zeros(shape=(len(index_array),) +
                           self.target_size + (self.num_classes,))

        for i, idx in enumerate(index_array):
            image, label = load_image(
                self.images_list[idx]), load_image(self.labels_list[idx])
            img_w, img_h = image.shape[0], image.shape[1]
            # do padding before random crop
            # if self.pad_size:
            #     pad_w = max(self.pad_size[1] - img_w, 0)
            #     pad_h = max(self.pad_size[0] - img_h, 0)
            # else:
            #     pad_w = max(self.target_size[1] - img_w, 0)
            #     pad_h = max(self.target_size[0] - img_h, 0)

            # image = np.lib.pad(image, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2,
            #                                                               pad_w - pad_w // 2), (0, 0)), 'constant', constant_values=0.)
            # label = np.lib.pad(label, ((pad_h // 2, pad_h - pad_h // 2), (pad_w // 2, pad_w -
            #                                                               pad_w // 2), (0, 0)), 'constant', constant_values=0.)
            # random crop
            if self.image_data_generator.random_crop:
                image, label = random_crop(image, label, self.target_size)
            else:
                image, label = resize_image(image, label, self.target_size)
            # data augmentation
            if np.random.uniform(0., 1.) < self.data_aug_rate:
                # random vertical flip
                if np.random.randint(2):
                    image, label = random_vertical_flip(
                        image, label, self.image_data_generator.vertical_flip)
                # random horizontal flip
                if np.random.randint(2):
                    image, label = random_horizontal_flip(
                        image, label, self.image_data_generator.horizontal_flip)
                # random brightness
                if np.random.randint(2):
                    image, label = random_brightness(
                        image, label, self.image_data_generator.brightness_range)
                # random rotation
                if np.random.randint(2):
                    image, label = random_rotation(
                        image, label, self.image_data_generator.rotation_range)
                # random channel shift
                if np.random.randint(2):
                    image, label = random_channel_shift(
                        image, label, self.image_data_generator.channel_shift_range)
                # random zoom
                if np.random.randint(2):
                    image, label = random_zoom(
                        image, label, self.image_data_generator.zoom_range)

            image = imagenet_utils.preprocess_input(image.astype('float32'), data_format='channels_last',
                                                    mode='torch')
            label = one_hot(label, self.num_classes)

            batch_x[i], batch_y[i] = image, label

        return batch_x, batch_y

    def next(self):
        """For python 2.x.

        # Returns
            The next batch.
        """
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch.
        with self.lock:
            index_array = next(self.index_generator)
        # The transformation of images is not under thread lock
        # so it can be done in parallel
        return self._get_batches_of_transformed_samples(index_array)


class ImageDataGenerator(object):
    def __init__(self,
                 random_crop=False,
                 rotation_range=0,
                 brightness_range=None,
                 zoom_range=0.0,
                 channel_shift_range=0.0,
                 horizontal_flip=False,
                 vertical_flip=False):
        self.random_crop = random_crop
        self.rotation_range = rotation_range
        self.brightness_range = brightness_range
        self.zoom_range = zoom_range
        self.channel_shift_range = channel_shift_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip

    def flow(self,
             images_list,
             labels_list,
             num_classes,
             batch_size,
             target_size,
             pad_size,
             shuffle=True,
             seed=None,
             data_aug_rate=0.):
        return DataIterator(image_data_generator=self,
                            images_list=images_list,
                            labels_list=labels_list,
                            num_classes=num_classes,
                            batch_size=batch_size,
                            target_size=target_size,
                            pad_size=pad_size,
                            shuffle=shuffle,
                            seed=seed,
                            data_aug_rate=data_aug_rate)

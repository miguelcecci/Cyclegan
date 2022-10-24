import tensorflow as tf

class Preprocess:

    def __init__(self, resolution=[256, 256], channels=3):

        self.BUFFER_SIZE = 1000
        self.HEIGHT = resolution[0]
        self.WIDTH = resolution[1]
        self.BATCH_SIZE = 1
        self.CHANNELS = channels
        self.AUTOTUNE = tf.data.AUTOTUNE



    def random_crop(self, image):
        cropped_image = tf.image.random_crop(
            image, size=[self.HEIGHT, self.WIDTH, self.CHANNELS])

        return cropped_image

    def normalize(self, image):
        image = tf.cast(image, tf.float32)
        image = (image / 127.5) - 1
        return image

    def augmentation(self, image):
        image = tf.image.resize(image, [int(self.HEIGHT*1.1), int(self.WIDTH*1.1)],
                            method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

#     image = tf.image.adjust_brightness(image, random.uniform(-.5, .5))
#     image = tf.image.adjust_contrast(image, random.uniform(0.7, 1.3))

        image = self.random_crop(image)

        image = tf.image.random_flip_left_right(image)

        return image

    def preprocess_image_train(self, image, label):
        image = self.augmentation(image)
        image = self.normalize(image)
        return image

    def preprocess_image_test(self, image, label):
        image = self.normalize(image)
        return image


    def preprocess_all(self, train_a, train_b, test_a=None, test_b=None):
        
        train_a = train_a.cache().map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        train_b = train_b.cache().map(
            self.preprocess_image_train, num_parallel_calls=self.AUTOTUNE).shuffle(
            self.BUFFER_SIZE).batch(self.BATCH_SIZE)

        if test_a != None and test_b != None:
            test_a = test_a.map(
                self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
                self.BUFFER_SIZE).batch(self.BATCH_SIZE)

            test_b = test_b.map(
                self.preprocess_image_test, num_parallel_calls=self.AUTOTUNE).cache().shuffle(
                self.BUFFER_SIZE).batch(self.BATCH_SIZE)
        
        return train_a, train_b, test_a, test_b

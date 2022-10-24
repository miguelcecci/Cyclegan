import tensorflow as tf
from tqdm import tqdm
import time
import cv2


class Cyclegan:

    def __init__(self, generator_a, generator_b, discriminator_a, discriminator_b):
        self.LAMBDA = 10
        self.learning_rate = 2e-4
        self.beta_1 = 0.5

        self.generator_a = generator_a
        self.generator_b = generator_b
        self.discriminator_a = discriminator_a
        self.discriminator_b = discriminator_b

        self.generator_a_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)
        self.generator_b_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)
        self.discriminator_a_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)
        self.discriminator_b_optimizer = tf.keras.optimizers.Adam(self.learning_rate, beta_1=self.beta_1)

        self.loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

        self.checkpoint_path = './checkpoints/train'
        self.checkpoint = tf.train.Checkpoint(
                generator_a = self.generator_a,
                generator_b = self.generator_b,
                discriminator_a = self.discriminator_a,
                discriminator_b = self.discriminator_b,
                generator_a_optimizer = self.generator_a_optimizer,
                generator_b_optimizer = self.generator_b_optimizer,
                discriminator_a_optimizer = self.discriminator_a_optimizer,
                discriminator_b_optimizer = self.discriminator_b_optimizer
                )
        self.checkpoint_manager = tf.train.CheckpointManager( self.checkpoint, self.checkpoint_path, max_to_keep=3)

    # losses


    def discriminator_loss(self, real, generated):
        real_loss = self.loss(tf.ones_like(real), real)
        generated_loss = self.loss(tf.zeros_like(generated), generated)

        total_discriminator_loss = real_loss + generated_loss

        return total_discriminator_loss * 0.5

    def generator_loss(self, generated):
        return self.loss(tf.ones_like(generated), generated)

    def cycle_loss(self, real_image, cycled_image):
        return self.LAMBDA*tf.reduce_mean(tf.abs(real_image - cycled_image))

    def identity_loss(self, real_image, same_image):
        return self.LAMBDA * 0.5 * tf.reduce_mean(tf.abs(real_image - same_image))

    
    def checkpoint_loader(self, path):
        
        """
        to do: set checkpoint manager to attribute
        """

        if self.checkpoint_manager.latest_checkpoint:
            print("Loading Checkpoint")
            self.checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print("Checkpoint restored")

    def checkpoint_saver(self):

        """
        to do
        """

        self.checkpoint_manager.save()

        pass

        

    def generate(self, model, input_image, save_path):
        """
        generate and save image
        """
        prediction = model(input_image)[0]

        prediction = np.float32(np.array(prediction).astype(int))
        prediction = cv2.cvtColor(prediction, cv2.COLOR_BGR2RGB)

        cv.imwrite(save_path, prediction)


    @tf.function
    def train_step(self, real_a, real_b):

        with tf.GradientTape(persistent=True) as tape:
            
            fake_b = self.generator_b(real_a, training=True)
            cycled_a = self.generator_a(fake_b, training=True)

            fake_a = self.generator_a(real_b, training=True)
            cycled_b = self.generator_b(fake_a, training=True)

            same_a = self.generator_a(real_a, training=True)
            same_b = self.generator_b(real_b, training=True)

            

            # discriminator

            disc_real_a = self.discriminator_a(real_a, training=True)
            disc_real_b = self.discriminator_b(real_b, training=True)

            disc_fake_a = self.discriminator_a(fake_a, training=True)
            disc_fake_b = self.discriminator_b(fake_b, training=True)

            #losses

            gen_b_loss = self.generator_loss(disc_fake_b)
            gen_a_loss = self.generator_loss(disc_fake_a)


            total_cycle_loss = self.cycle_loss(real_a, cycled_a) + self.cycle_loss(real_b, cycled_b)

            total_generator_b_loss = gen_b_loss + total_cycle_loss + self.identity_loss(real_b, same_b)
            total_generator_a_loss = gen_a_loss + total_cycle_loss + self.identity_loss(real_a, same_a)

            discriminator_a_loss = self.discriminator_loss(disc_real_a, disc_fake_a)
            discriminator_b_loss = self.discriminator_loss(disc_real_b, disc_fake_b)

        generator_a_gradients = tape.gradient(
                total_generator_a_loss, 
                self.generator_a.trainable_variables
                )

        generator_b_gradients = tape.gradient(
                total_generator_b_loss,
                self.generator_b.trainable_variables
                )

        discriminator_a_gradients = tape.gradient(
                discriminator_a_loss,
                self.discriminator_a.trainable_variables
                )

        discriminator_b_gradients = tape.gradient(
               discriminator_b_loss,
               self.discriminator_b.trainable_variables
               )

        self.generator_a_optimizer.apply_gradients(
                zip(
                    generator_a_gradients,
                    self.generator_a.trainable_variables
                    )
                )

        self.generator_b_optimizer.apply_gradients(
                zip(
                    generator_b_gradients,
                    self.generator_b.trainable_variables
                    )
                )

        self.discriminator_a_optimizer.apply_gradients(
                zip(
                    discriminator_a_gradients,
                    self.discriminator_a.trainable_variables
                    )
                )

        self.discriminator_b_optimizer.apply_gradients(
                zip(
                    discriminator_b_gradients,
                    self.discriminator_b.trainable_variables
                    )
                )

    def train_epoch(self, image_a, image_b):

        for image_a, image_b in tf.data.Dataset.zip((image_a, image_b)):
            self.train_step(image_a, image_b)

    def generate_images(model, test_input, save=False):
        prediction = model(test_input)
            
        plt.figure(figsize=(12, 12))

        display_list = [test_input[0], prediction[0]]
        title = ['Input Image', 'Predicted Image']

        for i in range(2):
            plt.subplot(1, 2, i+1)
            plt.title(title[i])
            # getting the pixel values between [0, 1] to plot it.
            plt.imshow(display_list[i] * 0.5 + 0.5)
            plt.axis('off')
        plt.show()


    def train_n_epochs(self, image_a, image_b, epochs, show_progress=True):

        print(f"Start training for {epochs} epochs")

        for i in tqdm(range(epochs)):
            self.train_epoch(image_a, image_b)








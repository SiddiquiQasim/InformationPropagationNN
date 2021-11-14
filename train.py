import tensorflow as tf
from Model import Model

def main():
    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(60000, 784).astype("float32") / 255
    x_test = x_test.reshape(10000, 784).astype("float32") / 255

    BATCH_SIZE = 32
    STEPS_PER_EPOCH = y_train.size / BATCH_SIZE
    SAVE_PERIOD = 1
    ACCURACY_THRESHOLD = 1.00
    save_weight = True

    if save_weight:
        saveWeights = tf.keras.callbacks.ModelCheckpoint('/parameter/w{sigma_w}b{sigma_b}/{epoch}.hdf5',
                                                    save_weights_only=True,
                                                    verbose = 0, save_freq = int (SAVE_PERIOD * STEPS_PER_EPOCH))

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') == ACCURACY_THRESHOLD):
                print("\nReached %2.2f%% accuracy, so stopping training!!" %(ACCURACY_THRESHOLD*100))
                self.model.stop_training = True
    # Instantiate a callback object
    stopping = myCallback()

    model = Model(sigma_w=2, sigma_b=0.25, maxL=100, Nl=1000, input_size=784, classe=10, learning_rate=0.001).build()
    if save_weight:
        r = model.fit(x_train, y_train, epochs=200,batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH,callbacks=[saveWeights, stopping])
    else:
        r = model.fit(x_train, y_train, epochs=200,batch_size=BATCH_SIZE,
                steps_per_epoch=STEPS_PER_EPOCH,callbacks=[stopping])


if __name__ == '__main__':
    main()
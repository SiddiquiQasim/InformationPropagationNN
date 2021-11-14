import tensorflow as tf

class Model:
    

    def __init__(self, sigma_w, sigma_b, maxL, Nl, input_size=784, classes=10, learning_rate=0.001):

        '''Gaussian initialized weights with mean=0, var=sigma_w and weigth scaling
        Gaussian initialized biases with mean=0 and var=sigma_b
        maxL = Number of Hidden layers
        Nl = Number of neurons in the hidden layers
        '''
        self.sigma_w = sigma_w
        self.sigma_b = sigma_b
        self.maxL = maxL
        self.Nl = Nl
        self.input_size = input_size
        self.classes = classes
        self.learning_rate = learning_rate

    def build(self):
       
        wi = tf.keras.initializers.VarianceScaling(scale=self.sigma_w, mode='fan_in', distribution='truncated_normal')
        bi = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=tf.sqrt(self.sigma_b))

        model = tf.keras.Sequential()
        model.add(tf.keras.Input(shape=(self.input_size,)))
        for i in range(0, self.maxL):
          model.add(tf.keras.layers.Dense(self.Nl, kernel_initializer = wi, bias_initializer = bi))
          model.add(tf.keras.layers.Activation(tf.nn.tanh))
        model.add(tf.keras.layers.Dense(self.classes, activation = 'softmax'))

        opt = tf.keras.optimizers.SGD(learning_rate=self.learning_rate, momentum=0.0, nesterov=False)

        model.compile(optimizer=opt,
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

        return model
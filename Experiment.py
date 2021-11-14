import tensorflow as tf
import numpy as np
from keras import backend as K
from Model import Model

class Experment:
    def __init__(self, sigma_w, sigma_b, maxL=100, Nl=1000, input_size=784, classes=10, learning_rate=0.001):
        '''Gaussian initialized weights with mean=0, var=sigma_w and weigth scaling
        Gaussian initialized biases with mean=0 and var=sigma_b
        maxL = Number of Hidden layers
        Nl = Number of neurons in the hidden layers
        '''
        self.model = Model(sigma_w, sigma_b, maxL, Nl, input_size, classes, learning_rate).build()


    def ql(self, l, x0, Nl):
        
        '''Computing the normalized square length of a single pre-activation vector on layer 'l' 
        l = layer at which we compute ql
        x0 = input value [1, input_size]
        Nl = Number of neurons in the layer
        '''
        get_layer_output = K.function([self.model.layers[0].input],
                                    [self.model.layers[l].output])
        layer_output = get_layer_output([x0])[0]

        return (tf.tensordot(layer_output ,layer_output, [[1],[1]])/Nl).eval(session=tf.compat.v1.Session()), layer_output

    def q_ab(self, l, x0a, x0b, Nl):

        get_layer_output = K.function([self.model.layers[0].input],
                                        [self.model.layers[l].output])
        layer_output1 = get_layer_output([x0a])[0]
        layer_output2 = get_layer_output([x0b])[0]

        q12 = tf.tensordot(layer_output1 ,layer_output2, [[1],[1]])/Nl
        q11 = tf.tensordot(layer_output1 ,layer_output1, [[1],[1]])/Nl
        q22 = tf.tensordot(layer_output2 ,layer_output2, [[1],[1]])/Nl

        return q12, q11, q22

    
    def cl(self, l, x0a, x0b, Nl):
        '''
        l = layer at which we compute q_ab
        x0a = first input value [1, input_size]
        x0b = second input value [1, input_size]
        Nl = Number of neurons in the layer
        '''
        qab, qa, qb = self.q_ab(l, Nl, x0a, x0b)
        
        return qab / tf.sqrt(tf.matmul(qa ,qb))

    def meanfield_init(self, Nl):
        '''
        Calculate the mean and stddev of the Initial parameters using mean field theory
        Nl = Number of neuron in hidden layers
        '''
        model = self.model
        mean_b = np.array([])
        var_b = np.array([])
        mean_w = np.array([])
        var_w = np.array([])

        for j in range(0,201,2):
            mean_b = np.append(mean_b, tf.math.reduce_mean(model.layers[j].get_weights()[1]))
            var_b = np.append(var_b, (tf.math.reduce_std(model.layers[j].get_weights()[1])**2))
            mean_w = np.append(mean_w, tf.math.reduce_mean(model.layers[j].get_weights()[0]))
            var_w = np.append(var_w, (tf.math.reduce_std(model.layers[j].get_weights()[0])**2)*Nl)

        return np.mean(mean_b), np.mean(var_b), np.mean(mean_w), np.mean(var_w)


    def meanfield_sw_sb(self, sigma_w, sigma_b, Nl):
        '''
        Calculate the mean and stddev of the parameters at ever epochs using mean field theory
        sigma_w = Initial variance of weigts
        sigma_b = Intinial variance of biases
        Nl = Number of neuron in hidden layers
        '''
        model = self.model

        meanfield_mb = np.array([0])
        meanfield_vb = np.array([sigma_b])
        meanfield_mw = np.array([0])
        meanfield_vw = np.array([sigma_w])

        for i in range(1, 60):
            mean_b = np.array([])
            var_b = np.array([])
            mean_w = np.array([])
            var_w = np.array([])
            model.load_weights('/parameter/w{sigma_w}b{sigma_b}/{i}.hdf5')

            for j in range(0,201,2):
                mean_b = np.append(mean_b, tf.math.reduce_mean(model.layers[j].get_weights()[1]))
                var_b = np.append(var_b, (tf.math.reduce_std(model.layers[j].get_weights()[1])**2))
                mean_w = np.append(mean_w, tf.math.reduce_mean(model.layers[j].get_weights()[0]))
                var_w = np.append(var_w, (tf.math.reduce_std(model.layers[j].get_weights()[0])**2)*Nl)
                
            meanfield_mb = np.append(meanfield_mb, np.mean(mean_b))
            meanfield_vb = np.append(meanfield_vb, np.mean(var_b))
            meanfield_mw = np.append(meanfield_mw, np.mean(mean_w))
            meanfield_vw = np.append(meanfield_vw, np.mean(var_w))

        return meanfield_mb, meanfield_vb, meanfield_mw, meanfield_vw
        

    def weightvector(self, i):

        '''
        Vectorizing all weights of the model
        '''
        model = self.model
        model.load_weights('/parameter/w2b0.25/{}.hdf5'.format(i))
        allweights = tf.reshape(tf.constant([model.layers[0].get_weights()[0]]), [784000])
        for j in range(2, 200, 2):
            allweights = tf.concat([allweights, tf.reshape(model.layers[j].get_weights()[0], [1000000])], 0)

        allweights = tf.concat([allweights, tf.reshape(model.layers[200].get_weights()[0], [10000])], 0)

        return allweights

    
    def euclideandist(self, start, end):

        '''
        Calculating euclidean distance btw adjacent epochs and
        sum the distance to find the total distance travel/changed during training 
        '''
        alldist = np.array([])
        for i in range(start, end-1):
            print(i)
            subt = tf.subtract(self.weightvector(i), self.weightvector(i+1))
            dist = tf.tensordot(subt, subt, 1)
            eucld = tf.sqrt(dist)
            alldist = np.append(alldist, eucld)

        totaltravel = np.sum(alldist)

        return totaltravel, alldist

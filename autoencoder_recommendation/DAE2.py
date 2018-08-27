import tensorflow as tf
import model_helper


class DAE:

    def __init__(self, learning_rate, _lambda, num_v, num_h,reg,num_layer,act_func):
        ''' Imlimentation of deep autoencoder class.'''
        self.num_layer = num_layer
        self.learning_rate = learning_rate
        self._lambda = _lambda
        self.num_v=num_v
        self.num_h = num_h
        self.reg = reg
        self.weight_initializer = model_helper._get_weight_initializer()
        self.bias_initializer = model_helper._get_bias_initializer()
        self.act_func = act_func
        self.init_parameters()


    def init_parameters(self):
        '''Initialize networks weights abd biasis.'''

        with tf.name_scope('weights'):  # name_scope make variables below have same name 'weights'
            self.weight_list = []
            self.W_0 = tf.get_variable(name='weight_0', shape=(self.num_v, self.num_h),
                                       initializer=self.weight_initializer)
            self.weight_list.append(self.W_0)
            for i in range(self.num_layer - 1):
                self.weight_list.append(
                    tf.get_variable(name='weight_' + str(i + 1), shape=(self.num_h, self.num_h),
                                    initializer=self.weight_initializer))
            # self.W_2=tf.get_variable(name='weight_2', shape=(self.FLAGS.num_h,self.FLAGS.num_h),
            #                          initializer=self.weight_initializer)
            # self.W_3=tf.get_variable(name='weight_3', shape=(self.FLAGS.num_h,self.FLAGS.num_h),
            #                          initializer=self.weight_initializer)
            self.W_last = tf.get_variable(name='weight_' + str(self.num_layer),
                                          shape=(self.num_h, self.num_v),
                                          initializer=self.weight_initializer)
            self.weight_list.append(self.W_last)

        with tf.name_scope('biases'):
            self.bias_list = []
            for i in range(self.num_layer):
                self.bias_list.append(tf.get_variable(name='bias_' + str(i), shape=(self.num_h),
                                                      initializer=self.bias_initializer))
            # self.b1=tf.get_variable(name='bias_1', shape=(self.FLAGS.num_h),
            #                         initializer=self.bias_initializer)
            # self.b2=tf.get_variable(name='bias_2', shape=(self.FLAGS.num_h),
            #                         initializer=self.bias_initializer)
            # self.b3=tf.get_variable(name='bias_3', shape=(self.FLAGS.num_h),
            #                         initializer=self.bias_initializer)

    def _inference(self, x):
        ''' Making one forward pass. Predicting the networks outputs.
        @param x: input ratings

        @return : networks predictions
        '''

        with tf.name_scope('inference'):
            temp = None
            for i in range(self.num_layer):
                if self.act_func == 'Sigmoid':
                    if i == 0:
                        temp = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.weight_list[0]), self.bias_list[0]))
                    else:
                        temp = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(temp, self.weight_list[i]), self.bias_list[i]))
                elif self.act_func =='Relu':
                    if i == 0:
                        temp = tf.nn.relu(tf.nn.bias_add(tf.matmul(x, self.weight_list[0]), self.bias_list[0]))
                    else:
                        temp = tf.nn.relu(tf.nn.bias_add(tf.matmul(temp, self.weight_list[i]), self.bias_list[i]))

            # a1=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x, self.W_1),self.b1))
            # a2=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a1, self.W_2),self.b2))
            # a3=tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(a2, self.W_3),self.b3))
            temp = tf.matmul(temp, self.weight_list[-1])
        return temp

    def _compute_loss(self, predictions, labels, num_labels):
        ''' Computing the Mean Squared Error loss between the input and output of the network.

    	  @param predictions: predictions of the stacked autoencoder
    	  @param labels: input values of the stacked autoencoder which serve as labels at the same time
    	  @param num_labels: number of labels !=0 in the data set to compute the mean

    	  @return mean squared error loss tf-operation
    	  '''

        with tf.name_scope('loss'):
            loss_op = tf.div(tf.reduce_sum(tf.square(tf.subtract(predictions, labels))), num_labels)
            return loss_op

    def _optimizer(self, x):
        '''Optimization of the network parameter through stochastic gradient descent.

            @param x: input values for the stacked autoencoder.

            @return: tensorflow training operation
            @return: ROOT!! mean squared error
        '''

        outputs = self._inference(x)
        mask = tf.where(tf.equal(x, 0.0), tf.zeros_like(x),
                        x)  # indices of 0 values in the training set(https://www.tensorflow.org/api_docs/python/tf/where)
        # The condition tensor acts as a mask that chooses, based on the value at each element,
        # whether the corresponding element / row in the output should be taken from x (if true) or y (if false)
        num_train_labels = tf.cast(tf.count_nonzero(mask),
                                   dtype=tf.float32)  # number of non zero values in the training set
        bool_mask = tf.cast(mask, dtype=tf.bool)  # boolean mask
        outputs = tf.where(bool_mask, outputs, tf.zeros_like(
            outputs))  # set the output values to zero if corresponding input values are zero

        MSE_loss = self._compute_loss(outputs, x, num_train_labels)
        # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        # MSE_loss = MSE_loss + self._lambda * l2_loss
        if self.reg == 'L2':
            l2_loss = 0
            for item in self.weight_list:
                l2_loss+= tf.contrib.layers.l2_regularizer(self._lambda)(item)
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            MSE_loss = MSE_loss + self._lambda * l2_loss
        elif self.reg == 'L1':
            l1_loss = 0
            for item in self.weight_list:
                l1_loss+= tf.contrib.layers.l1_regularizer(self._lambda)(item)
            # l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            MSE_loss = MSE_loss + self._lambda * l1_loss
        train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(MSE_loss)
        RMSE_loss = tf.sqrt(MSE_loss)

        return train_op, RMSE_loss

    def _validation_loss(self, x_train, x_test):

        ''' Computing the loss during the validation time.

    	  @param x_train: training data samples
    	  @param x_test: test data samples

    	  @return networks predictions
    	  @return root mean squared error loss between the predicted and actual ratings
    	  '''

        outputs = self._inference(x_train)  # use training sample to make prediction
        mask = tf.where(tf.equal(x_test, 0.0), tf.zeros_like(x_test),
                        x_test)  # identify the zero values in the test ste
        num_test_labels = tf.cast(tf.count_nonzero(mask), dtype=tf.float32)  # count the number of non zero values
        bool_mask = tf.cast(mask, dtype=tf.bool)
        outputs = tf.where(bool_mask, outputs, tf.zeros_like(outputs))

        MSE_loss = self._compute_loss(outputs, x_test, num_test_labels)
        RMSE_loss = tf.sqrt(MSE_loss)

        return outputs, RMSE_loss







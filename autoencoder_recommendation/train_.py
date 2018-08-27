import tensorflow as tf
from data.dataset import _get_training_data2, _get_test_data2
from DAE2 import DAE

class train(object):
    def __init__(self,num_layer=5, reg='L1', act_func='Relu', batch_norm = False,num_epoch = 31):
        self.train_record_path = './ml-1m/train/'
        self.test_record_path = './ml-1m/test/'
        self.num_epoch = num_epoch
        self.batch_size=16
        self.learning_rate = 0.0005
        self._lambda = 0.01
        self.num_v=3952
        self.num_h = 128
        self.num_samples = 5953
        self.num_layer = num_layer
        self.reg = reg
        self.act_func = act_func
        self.batch_norm = batch_norm

    def normalize(self,data,num):
        axis=list(range(len(data.get_shape()) - 1))
        mean, variance = tf.nn.moments(data, axis)
        size = data.get_shape().as_list()[1]
        scale = tf.get_variable('scale'+str(num), [size], initializer=tf.constant_initializer(0.1))
        offset = tf.get_variable('offset'+str(num), [size])
        return tf.nn.batch_normalization(data, mean, variance,offset,scale,0.01)
    def train(self, _):
        num_batches = int(self.num_samples / self.batch_size)

        with tf.Graph().as_default():

            train_data, train_data_infer=_get_training_data2(self.train_record_path, self.batch_size)#get TFRecordDataset
            test_data=_get_test_data2(self.test_record_path)

            iter_train = train_data.make_initializable_iterator()
            iter_train_infer=train_data_infer.make_initializable_iterator()
            iter_test=test_data.make_initializable_iterator()

            x_train= iter_train.get_next()#one batch?
            x_train_infer=iter_train_infer.get_next()
            x_test=iter_test.get_next()
            if self.batch_norm:
                x_train, x_train_infer, x_test = self.normalize(x_train,1),self.normalize(x_train_infer,2),self.normalize(x_test,3)

            model=DAE(self.learning_rate, self._lambda, self.num_v,self.num_h, self.reg,self.num_layer,self.act_func)#model part

            train_op, train_loss_op=model._optimizer(x_train)
            pred_op, test_loss_op=model._validation_loss(x_train_infer, x_test)


            with tf.Session() as sess:

                sess.run(tf.global_variables_initializer())
                train_loss=0
                test_loss=0
                res = []
                for epoch in range(self.num_epoch):

                    sess.run(iter_train.initializer)

                    for batch_nr in range(num_batches):

                        _, loss_=sess.run((train_op, train_loss_op))
                        train_loss+=loss_

                    sess.run(iter_train_infer.initializer)
                    sess.run(iter_test.initializer)

                    for i in range(self.num_samples):
                        pred, loss_=sess.run((pred_op, test_loss_op))
                        test_loss+=loss_
                    if epoch%5==0:
                        print('num_layer: %i, epoch_nr: %i, train_loss: %.3f, test_loss: %.3f'%(self.num_layer,epoch, (train_loss/num_batches),(test_loss/self.num_samples)))
                        res.append((epoch,(train_loss/num_batches),(test_loss/self.num_samples)))
                    train_loss=0
                    test_loss=0
                return res

if __name__ == "__main__":
    t = train()
    t.train(None)
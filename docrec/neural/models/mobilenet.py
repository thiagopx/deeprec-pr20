import os
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.applications import MobileNetV2
from keras.layers import Conv2D, Input, ReLU, GlobalAveragePooling2D
from keras.initializers import RandomNormal


# https://github.com/keras-team/keras/issues/8090
class MobileNetFC:

    def __init__(self, input_tensor, num_classes=1000, sess=None):
        '''
        Reference:
        @article{howard2017mobilenets,
        title={Mobilenets: Efficient convolutional neural networks for mobile vision applications},
        author={Howard, Andrew G and Zhu, Menglong and Chen, Bo and Kalenichenko, Dmitry and Wang, Weijun and Weyand, Tobias and Andreetto, Marco and Adam, Hartwig},
        journal={arXiv preprint arXiv:1704.04861},
        year={2017}
        }
        '''

        self.sess = sess    
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
        K.set_session(self.sess)

        self.base_model = MobileNetV2(input_tensor=input_tensor, include_top=False, weights=None)
        x = Conv2D(
            num_classes, kernel_size=1, name='Conv_Class',
            kernel_initializer=RandomNormal(mean=0.0, stddev=0.01),
            activation='relu'
        )(self.base_model.output)
        x = GlobalAveragePooling2D()(x)
        self.model = Model(inputs=input_tensor, outputs=x)
        self.output = self.model.output


    def load_base_weights(self, weights_path):

        self.base_model.load_weights(weights_path)

    def load_weights(self, weights_path):

        self.model.load_weights(weights_path)

    def save_weights(self, weights_path):

        self.model.save_weights(weights_path)


# http://zachmoshe.com/2017/11/11/use-keras-models-with-tf.html
if __name__ == '__main__':
    import json
    from keras.models import model_from_json
    from keras import backend as K

    np.random.seed(0)
    tf.set_random_seed(0) # <= change this in case of multiple runs
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    size = 32
    X = 2 * np.random.random((5, size, size, 3)) - 1
    img_ph = tf.placeholder(dtype=tf.float32, shape=(None, size, size, 3))
    #base_model = MobileNetV2(input_tensor=img_ph, include_top=False, weights=None)
    #new_model = MobileNetFC(input_shape=(32, 32, 3), num_classes=2)

    # m1
    sess = tf.Session(config=config)
    K.set_session(sess)
    model = MobileNetFC(input_tensor=img_ph, num_classes=2)
    logits_op = model.output
    writer = tf.summary.FileWriter('/tmp/tensorboard', sess.graph)
    sess.run(tf.global_variables_initializer())
    model.load_base_weights('docrec/neural/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5')
    logits = sess.run(logits_op, feed_dict={img_ph: X})
    print(logits)
    sess.close()

    #model = MobileNetV2(input_tensor=img_ph, include_top=False, pooling='avg', weights='imagenet')
    #sess.run(tf.local_variables_initializer())
    #sess.run(tf.global_variables_initializer())
    #not_init_vars_names = sess.run(tf.report_uninitialized_variables())
    #not_init_vars = [var for var in tf.global_variables() if var.name not in not_init_vars_names]
    #sess.run(tf.variables_initializer(not_init_vars))
    #model.load_weights('docrec/neural/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    # model.load_base_weights('docrec/neural/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    # y = sess.run(model(img_ph), feed_dict={img_ph: X})
    # print(y, y.shape)

    # base_model.load_weights('docrec/neural/models/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5')
    # x, y = sess.run([base_model(img_ph), model.base_model(img_ph)], feed_dict={img_ph: X})
    # print(x - y)

    # model.save_weights('/tmp/model.h5')

    # new_model = MobileNetFC(input_shape=(32, 32, 3), num_classes=2)
    # #sess.run(tf.global_variables_initializer())
    # new_model.load_weights('/tmp/model.h5')
    # x, y = sess.run([model(img_ph), new_model(img_ph)], feed_dict={img_ph: X})
    # print(x - y)
    # sess.close()

        # np.random.seed(0)
        # X = 2*np.random.random((10, 32, 32, 3)) - 1
        # img_ph = tf.placeholder(dtype=tf.float32, shape=(None, 32, 32, 3))
        # model = mobilenet(input_layer=img_ph, num_classes=2, sess=sess, weights=None, seed=0)
        # not_init_vars_names = sess.run(tf.report_uninitialized_variables())
        # not_init_vars = [var for var in tf.global_variables() if var.name not in not_init_vars_names]
        # #print(not_init_vars)
        # #print(tf.global_variables())
        # #sess.run(tf.variables_initializer(not_init_vars))
        # sess.run(tf.variables_initializer(not_init_vars))
        # y = sess.run(model(img_ph), feed_dict={img_ph: X})
        # print(y)
        # layer = model.layers[1]
        # print(layer)
        # print(layer.get_weights())
        # # save model/weights
        # json.dump(model.to_json(), open('/tmp/model.json', 'w'))
        # model.save_weights('/tmp/model.h5')
        # '''
        # print(model.trainable_weights)
        # init_vars = []
        # for var in tf.global_variables():
        #     if var not in model.trainable_weights:
        #         init_vars.append(var)
        # print('------------------')
        # print(init_vars)

        # not_init = sess.run(tf.report_uninitialized_variables())
        # for var in model.trainable_weights:
        #     if var in not_init:
        #         print('trainable', var)

        # print('not init', len(not_init), len(init_vars))
        # '''
        # #print(set(tf.global_variables()) - set(model.trainable_weights))

        # # load model/weights
        # model_ = model_from_json(json.load(open('/tmp/model.json', 'r')))
        # model_.load_weights('/tmp/model.h5')
        # y = sess.run(model_(img_ph), feed_dict={img_ph: X})
        # print(y)

        # # padding test
        # paddings = tf.constant([[1, 0,], [1, 0]])
        # padded = tf.pad(model_(img_ph), paddings, 'CONSTANT')
        # y = sess.run(padded, feed_dict={img_ph: X})
        # print(y)

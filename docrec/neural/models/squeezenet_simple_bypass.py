
import os
import numpy as np
import tensorflow as tf


class SqueezeNetSB:

    def __init__(self, input_tensor, num_classes=1000, mode='train', model_scope='SqueezeNet', channels_first=False, sess=None, seed=0):
        ''' SqueezeNet v1.1 (simple bypass)
        Adpated from the Caffe original implementation: https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1
        Reference:
        @article{iandola2016squeezenet,
            title={Squeezenet: Alexnet-level accuracy with 50x fewer parameters and< 0.5 mb model size},
            author={Iandola, Forrest N and Han, Song and Moskewicz, Matthew W and Ashraf, Khalid and Dally, William J and Keutzer, Kurt},
            journal={arXiv preprint arXiv:1602.07360},
            year={2016}
        }
                         MaxPool    Bypass
        SqueezeNet 1.0 | 1, 4, 8 | 3, 5, 9
        Squeezenet 1.1 | 1, 3, 5 | 3, 4, 6
        '''

        assert mode in ['train', 'val', 'test']

        self.sess = sess
        if self.sess is None:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)

        self.scope = model_scope

        data_format = 'channels_first' if channels_first else 'channels_last'
        concat_axis = 1 if channels_first else 3
        with tf.variable_scope(model_scope, reuse=tf.AUTO_REUSE):
            conv1 = tf.layers.conv2d(input_tensor, 64, 3, 2, padding='valid', activation=tf.nn.relu, data_format=data_format, name='conv1')
            pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=3, strides=2, data_format=data_format)
            fire2_squeeze1x1 = tf.layers.conv2d(pool1, 16, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_squeeze1x1')
            fire2_expand1x1 = tf.layers.conv2d(fire2_squeeze1x1, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire2_expand1x1')
            fire2_expand3x3 = tf.layers.conv2d(fire2_squeeze1x1, 64, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire2_expand3x3')
            fire2_concat = tf.concat([fire2_expand1x1, fire2_expand3x3], axis=concat_axis)
            fire3_squeeze1x1 = tf.layers.conv2d(fire2_concat, 16, 1, activation=tf.nn.relu, data_format=data_format, name='fire3_squeeze1x1')
            fire3_expand1x1 = tf.layers.conv2d(fire3_squeeze1x1, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire3_expand1x1')
            fire3_expand3x3 = tf.layers.conv2d(fire3_squeeze1x1, 64, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire3_expand3x3')
            fire3_concat = tf.concat([fire3_expand1x1, fire3_expand3x3], axis=concat_axis)
            
            # bypass fire3
            bypass_fire3 = fire2_concat + fire3_concat
            pool3 = tf.layers.max_pooling2d(bypass_fire3, 3, 2, data_format=data_format)
            
            fire4_squeeze1x1 = tf.layers.conv2d(pool3, 32, 1, activation=tf.nn.relu, data_format=data_format, name='fire4_squeeze1x1')
            fire4_expand1x1 = tf.layers.conv2d(fire4_squeeze1x1, 128, 1, activation=tf.nn.relu, data_format=data_format, name='fire4_expand1x1')
            fire4_expand3x3 = tf.layers.conv2d(fire4_squeeze1x1, 128, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire4_expand3x3')
            fire4_concat = tf.concat([fire4_expand1x1, fire4_expand3x3], axis=concat_axis)
            fire5_squeeze1x1 = tf.layers.conv2d(fire4_concat, 32, 1, activation=tf.nn.relu, data_format=data_format, name='fire5_squeeze1x1')
            fire5_expand1x1 = tf.layers.conv2d(fire5_squeeze1x1, 128, 1, activation=tf.nn.relu, data_format=data_format, name='fire5_expand1x1')
            fire5_expand3x3 = tf.layers.conv2d(fire5_squeeze1x1, 128, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire5_expand3x3')
            fire5_concat = tf.concat([fire5_expand1x1, fire5_expand3x3], axis=concat_axis)
            
            # bypass fire5
            bypass_fire5 = fire4_concat + fire5_concat
            pool5 = tf.layers.max_pooling2d(bypass_fire5, 3, 2, data_format=data_format)
            
            fire6_squeeze1x1 = tf.layers.conv2d(pool5, 48, 1, activation=tf.nn.relu, data_format=data_format, name='fire6_squeeze1x1')
            fire6_expand1x1 = tf.layers.conv2d(fire6_squeeze1x1, 192, 1, activation=tf.nn.relu, data_format=data_format, name='fire6_expand1x1')
            fire6_expand3x3 = tf.layers.conv2d(fire6_squeeze1x1, 192, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire6_expand3x3')
            fire6_concat = tf.concat([fire6_expand1x1, fire6_expand3x3], axis=concat_axis)
            fire7_squeeze1x1 = tf.layers.conv2d(fire6_concat, 48, 1, activation=tf.nn.relu, data_format=data_format, name='fire7_squeeze1x1')
            fire7_expand1x1 = tf.layers.conv2d(fire7_squeeze1x1, 192, 1, activation=tf.nn.relu, data_format=data_format, name='fire7_expand1x1')
            fire7_expand3x3 = tf.layers.conv2d(fire7_squeeze1x1, 192, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire7_expand3x3')
            fire7_concat = tf.concat([fire7_expand1x1, fire7_expand3x3], axis=concat_axis)
            
            # bypass fire7
            bypass_fire7 = fire6_concat + fire7_concat
            
            fire8_squeeze1x1 = tf.layers.conv2d(bypass_fire7, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire8_squeeze1x1')
            fire8_expand1x1 = tf.layers.conv2d(fire8_squeeze1x1, 256, 1, activation=tf.nn.relu, data_format=data_format, name= 'fire8_expand1x1')
            fire8_expand3x3 = tf.layers.conv2d(fire8_squeeze1x1, 256, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire8_expand3x3')
            fire8_concat = tf.concat([fire8_expand1x1, fire8_expand3x3], axis=concat_axis)
            fire9_squeeze1x1 = tf.layers.conv2d(fire8_concat, 64, 1, activation=tf.nn.relu, data_format=data_format, name='fire9_squeeze1x1')
            fire9_expand1x1 = tf.layers.conv2d(fire9_squeeze1x1, 256, 1, activation=tf.nn.relu, data_format=data_format, name='fire9_expand1x1')
            fire9_expand3x3 = tf.layers.conv2d(fire9_squeeze1x1, 256, 3, padding='same', activation=tf.nn.relu, data_format=data_format, name='fire9_expand3x3')
            fire9_concat = tf.concat([fire9_expand1x1, fire9_expand3x3], axis=concat_axis)
            
            # bypass fire9
            bypass_fire9 = fire8_concat + fire9_concat
            
            drop9 = tf.layers.dropout(bypass_fire9, 0.5, training=(mode=='train'), seed=seed)

            conv10 = tf.layers.conv2d(
                drop9, num_classes, 1, kernel_initializer=tf.random_normal_initializer(0.0, 0.01, seed=seed),
                activation=tf.nn.relu, data_format=data_format, name='conv10'
            ) # discarded in case of finetuning with less than 1000 classes
            axes = [2, 3] if channels_first else [1, 2]
            logits = tf.reduce_mean(conv10, axes, keepdims=False, name='pool10')

            self.output = logits
            self.view = conv10


    def load_weights(self, weights_path, ignore_layers=[], BGR=False, ignore_missing=False):
        ''' Load network weights and biases (format caffe-tensorflow).
        data_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        first_layer: model first layer will be changed in case of BGR data.
        ignore_layers: layers whose parameters must be ignored.
        BGR: if data is BGR, convert weights from the first layer to RGB.
        ignore_missing: if true, serialized weights for missing layers are ignored.
        '''

        first_layer='conv1'
        data_dict = np.load(weights_path, encoding='latin1').item()
        for layer in data_dict:
            if layer in ignore_layers:
                continue
            for param_name, data in data_dict[layer].items():
                param_name = param_name.replace('weights', 'kernel').replace('biases', 'bias')
                try:
                    scope = '{}/{}'.format(self.scope, layer) if self.scope else layer
                    with tf.variable_scope(scope, reuse=True):
                        var = tf.get_variable(param_name)
                        if (layer == first_layer) and BGR and (param_name == 'kernel'):
                            data = data[:, :, [2, 1, 0], :] # BGR => RGB
                        self.sess.run(var.assign(data))
                except ValueError:
                    if not ignore_missing:
                        raise


    def save_weights(self, weights_path, ignore_layers=[]):
        ''' Load network weights and biases (format caffe-tensorflow).
        data_path: path to the numpy-serialized network weights.
        session: current TensorFlow session.
        ignore_layers: layers whose parameters must be ignored.
        '''

        data_dict = {}
        for var in tf.trainable_variables():
            layer, param_name = var.op.name.split('/')[-2 :] # excluce scope if existing
            if layer in ignore_layers:
                continue
            data = self.sess.run(var)
            try:
                data_dict[layer][param_name] = data
            except KeyError:
                data_dict[layer] = {param_name: data}

        # ckeck directory path
        if not os.path.exists(os.path.dirname(weights_path)):
            os.makedirs(os.path.dirname(weights_path))
        np.save(weights_path, np.array(data_dict))

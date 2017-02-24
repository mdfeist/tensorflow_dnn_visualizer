import os
import numpy as np
import scipy.misc
import cv2
import tensorflow as tf
import json

def _save_variable(path, variable, session):
    var = variable.value().eval(None, session)

    path = path + '.json'
    with open(path, 'w') as outfile:
        json.dump(var.tolist(), outfile)

    return path

#    if len(var.shape) == 4 and (var.shape[2] == 1 or var.shape[2] == 3):
#        data = var.transpose(3,0,1,2)
#        data = data.astype(np.float32)
#
#        # Normalize
#        minimum = data.flatten().min()
#        data = data - minimum
#        maximum = data.flatten().max()
#        data = data / maximum
#
#        n = int(np.ceil(np.sqrt(data.shape[0])))
#
#        padding = ((0, n ** 2 - data.shape[0]), (0, 0),
#                (0, 0)) + ((0, 0),) * (data.ndim - 3)
#        data = np.pad(data, padding, mode='constant',
#                constant_values=0)
#        # Tile the individual thumbnails into an image.
#        data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3)
#                + tuple(range(4, data.ndim + 1)))
#        data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
#        data = (data * 255.0).astype(np.uint8)
#
#        cv2.imwrite(path + '.png', data)
#        return path + '.png'
#    else:
#        path = path + '.json'
#        with open(path, 'w') as outfile:
#            json.dump(var.tolist(), outfile)
#
#        return path

class Visualizer_Layer:
    def __init__(
                self,
                tf_layer,
                scope='Layer',
                name=None):
        self.tf_layer = tf_layer
        self.scope = scope
        if name is None:
            self.name = scope
        else:
            self.name = name

class Visualizer_Saver:
    def __init__(self, logdir):
        self._logdir = logdir + 'visualizer/'
        self._layers = []

        os.mkdir(logdir)
        os.mkdir(self._logdir)

    def add_layer(self, layer):
        self._layers.append(layer)

    def save_network(self, session):
        json_data = {}
        json_data['layers'] = []
        for layer in self._layers:
            json_layer = {}
            json_layer['name'] = layer.name
            json_layer['vars'] = []

            #print layer.name
            #print layer.scope
            #print layer.tf_layer

            variables = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=layer.scope)

            for var in variables:
                json_var = {}
                json_var['name'] = str(var.name)
                json_var['dtype'] = str(var.dtype)
                json_var['shape'] = var.get_shape().as_list()

#                name = str(var.name).replace('/', '_').replace(':', '-')
#                path = os.path.join(self._logdir, 'layer_%s' % (name))
#
#                data_path = _save_variable(
#                    path,
#                    var,
#                    session)
#
#                json_var['data'] = data_path

                json_layer['vars'].append(json_var)

            json_data['layers'].append(json_layer)

        path = os.path.join(self._logdir, 'network.json')
        with open(path, 'w') as outfile:
            json.dump(json.JSONEncoder().encode(json_data), outfile)

    def save_activations(self, session, feed_dict):
        json_data = {}
        json_data['layers'] = []

        for layer in self._layers:
            json_layer = {}
            json_layer['name'] = layer.name

            output = session.run(layer.tf_layer, feed_dict=feed_dict)

            print layer.tf_layer
            print output

            json_data['layers'].append(json_layer)

        path = os.path.join(self._logdir, 'output.json')
        with open(path, 'w') as outfile:
            json.dump(json.JSONEncoder().encode(json_data), outfile)

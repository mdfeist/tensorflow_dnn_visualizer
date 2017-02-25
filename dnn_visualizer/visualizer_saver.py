import os
import numpy as np
import scipy.misc
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

    def save_activations(self, session, feed_dict, x, y):
        json_data = {}
        json_data['layers'] = []

        for layer in self._layers:
            json_layer = {}
            json_layer['name'] = layer.name
            json_layer['activations'] = []

            output = session.run(layer.tf_layer, feed_dict=feed_dict)

            print layer.tf_layer
            print output.shape

            json_layer['shape'] = output.shape

            if len(output.shape) == 4:

                data = output.transpose(0,3,1,2)
                data = data.astype(np.float32)

                # Normalize
                minimum = data.flatten().min()
                data = data - minimum
                maximum = data.flatten().max()
                data = data / maximum

                n = int(np.ceil(np.sqrt(data.shape[1])))

                padding = ((0, 0), (0, n ** 2 - data.shape[1]), (0, 0), (0, 0),)
                data = np.pad(data, padding, mode='constant',
                        constant_values=0)

                # Tile the individual thumbnails into an image.
                data = data.reshape((data.shape[0], n, n) + data.shape[2:])
                data = data.transpose(0, 1, 3, 2, 4)
                data = data.reshape((data.shape[0], n * data.shape[2], n * data.shape[4]))
                data = (data * 255.0).astype(np.uint8)
                for i in range(data.shape[0]):
                    json_layer_activations = {}

                    name = str(layer.name).replace('/', '_').replace(':', '-')
                    path = os.path.join(
                        self._logdir,
                        'layer_output_l%d_%s.png' % (i, name))

                    scipy.misc.imsave(path, data[i])

                    json_layer_activations['input'] = i
                    json_layer_activations['path'] = path
                    json_layer_activations['image'] = True

                    json_layer['activations'].append(json_layer_activations)
#            else:
#                path = path + '.json'
#                with open(path, 'w') as outfile:
#                    json.dump(var.tolist(), outfile)

            json_data['layers'].append(json_layer)

        path = os.path.join(self._logdir, 'output.json')
        with open(path, 'w') as outfile:
            json.dump(json.JSONEncoder().encode(json_data), outfile)

import numpy as np
import random
import gc
from sklearn.metrics.classification import accuracy_score
from sklearn.cross_decomposition import PLSRegression
import sys
import keras
from keras.layers import *
from keras.models import Model
from keras.regularizers import *
import argparse
from keras.callbacks import Callback
class LearningRateScheduler(Callback):

    def __init__(self, init_lr=0.01, schedule=[(25, 1e-2), (50, 1e-3), (100, 1e-4)]):
        super(Callback, self).__init__()
        self.init_lr = init_lr
        self.schedule = schedule

    def on_epoch_end(self, epoch, logs={}):
        lr = self.init_lr
        for i in range(0, len(self.schedule) - 1):
            if epoch >= self.schedule[i][0] and epoch < self.schedule[i + 1][0]:
                lr = self.schedule[i][1]

        if epoch >= self.schedule[-1][0]:
            lr = self.schedule[-1][1]

        print('Learning rate:{}'.format(lr))
        keras.backend.set_value(self.model.optimizer.lr, lr)

def save_model(file_name='', model=None):
    import keras
    print('Salving architecture and weights in {}'.format(file_name))

    model.save_weights(file_name + '.h5')
    with open(file_name + '.json', 'w') as f:
        f.write(model.to_json())

def compute_flops(model):
    import keras
    total_flops =0
    flops_per_layer = []

    try:
        layer = model.get_layer(index=1).layers #Just for discover the model type
        for layer_idx in range(1, len(model.get_layer(index=1).layers)):
            layer = model.get_layer(index=1).get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)
    except:
        for layer_idx in range(1, len(model.layers)):
            layer = model.get_layer(index=layer_idx)
            if isinstance(layer, keras.layers.Conv2D) is True:
                _, output_map_H, output_map_W, current_layer_depth = layer.output_shape

                _, _, _, previous_layer_depth = layer.input_shape
                kernel_H, kernel_W = layer.kernel_size

                flops = output_map_H * output_map_W * previous_layer_depth * current_layer_depth * kernel_H * kernel_W
                total_flops += flops
                flops_per_layer.append(flops)

            if isinstance(layer, keras.layers.Dense) is True:
                _, current_layer_depth = layer.output_shape

                _, previous_layer_depth = layer.input_shape

                flops = current_layer_depth * previous_layer_depth
                total_flops += flops
                flops_per_layer.append(flops)

    return total_flops, flops_per_layer

def resnet_layer(inputs,
                 num_filters=16,
                 kernel_size=3,
                 strides=1,
                 activation='relu',
                 batch_normalization=True,
                 conv_first=True,
                 name=''):

    conv = Conv2D(num_filters,
                  kernel_size=kernel_size,
                  strides=strides,
                  padding='same',
                  kernel_initializer='he_normal',
                  kernel_regularizer=l2(1e-4),
                  name='Conv2D_{}'.format(name))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm1_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act1_{}'.format(name))(x)
    else:
        if batch_normalization:
            x = BatchNormalization(name='BatchNorm2_{}'.format(name))(x)
        if activation is not None:
            x = Activation(activation, name='Act2_{}'.format(name))(x)
        x = conv(x)
    return x


def create_model(input_shape, depth_block,
                 iter=0, num_classes=10):
    num_filters = 16

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)
    # Instantiate the stack of residual units
    for stack in range(3):
        num_res_blocks = depth_block[stack]
        for res_block in range(num_res_blocks):
            layer_name = str(stack)+'_'+str(res_block)+'_'+str(iter)
            strides = 1
            if stack > 0 and res_block == 0:  # first layer but not first stack
                strides = 2  # downsample
            y = resnet_layer(inputs=x,
                             num_filters=num_filters,
                             strides=strides,
                             name=layer_name+'_1')
            y = resnet_layer(inputs=y,
                             num_filters=num_filters,
                             activation=None,
                             name=layer_name+'_2')
            if stack > 0 and res_block == 0:  # first layer but not first stack
                # linear projection residual shortcut connection to match
                # changed dims
                x = resnet_layer(inputs=x,
                                 num_filters=num_filters,
                                 kernel_size=1,
                                 strides=strides,
                                 activation=None,
                                 batch_normalization=False,
                                 name=layer_name+'_3')
            x = keras.layers.add([x, y],
                                 name='Add_'+layer_name)
            x = Activation('relu', name='Actoutput'+layer_name)(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection-ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

def vip(model):
    t = model.x_scores_  # (n_samples, n_components)
    w = model.x_weights_  # (p, n_components)
    q = model.y_loadings_  # (q, n_components)
    p, h = w.shape
    s = np.diag(t.T @ t @ q.T @ q).reshape(h, -1)
    w_norm = np.linalg.norm(w, axis=0)
    weights = (w / np.expand_dims(w_norm, axis=0)) ** 2
    return np.sqrt(p * (weights @ s).ravel() / np.sum(s))

def score(model=None, X_train=None, y_train=None, n_components=2, layers=[], pool_size=8):

    ranked = []
    for i in range(0, len(layers)):
        #Prepare to extract features from ith layer
        layer = model.get_layer(index=layers[i]).output
        output = Flatten()((AveragePooling2D(pool_size=pool_size)(layer)))
        model_layer = Model(model.input, output)
        X_train_layer = model_layer.predict(X_train, batch_size=32)


        dm = PLSRegression(n_components=n_components)
        dm.fit(X_train_layer, y_train)
        scores = vip(dm)
        del dm

        ranked.append(np.mean(scores))

        del model_layer
        gc.collect()

    return ranked

def random_crop(img=None, random_crop_size=(64, 64)):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y+dy), x:(x+dx), :]

def data_augmentation(X, padding=4):

    X_out = np.zeros(X.shape, dtype=X.dtype)
    n_samples, x, y, _ = X.shape

    padded_sample = np.zeros((x+padding*2, y+padding*2, 3), dtype=X.dtype)

    for i in range(0, n_samples):
        p = random.random()
        padded_sample[padding:x+padding, padding:y+padding, :] = X[i][:, :, :]
        if p >= 0.5: #random crop on the original image
            X_out[i] = random_crop(padded_sample, (x, y))
        else: #random crop on the flipped image
            X_out[i] = random_crop(np.flip(padded_sample, axis=1), (x, y))

    return X_out

def count_res_blocks(model, dim_block=[16, 32, 64]):
    #Returns the last Add of each block
    res_blocks = [0, 0, 0]

    for i in range(0, len(model.layers)-1):

        layer = model.get_layer(index=i)
        if isinstance(layer, Add):
            dim = layer.output.shape.as_list()[-1]
            if dim == 16:
                res_blocks[0] = res_blocks[0] + 1
            if dim == 32:
                res_blocks[1] = res_blocks[1] + 1
            if dim == 64:
                res_blocks[2] = res_blocks[2] + 1

    return res_blocks

def idx_blocks(model=None):
    #Returns the last Add of each block
    blocks = []
    for i in range(0, len(model.layers)-1):
        dim_next = None
        dim_prev = None

        layer_prev = model.get_layer(index=i)
        layer_next = model.get_layer(index=i+1)

        if isinstance(layer_prev, Activation) or isinstance(layer_prev, Conv2D):
            dim_prev = layer_prev.output.shape.as_list()[-1]

        if isinstance(layer_next, Activation) or isinstance(layer_next, Conv2D):
            dim_next = layer_next.output.shape.as_list()[-1]

        if (dim_prev !=None and dim_next != None) and (dim_next != dim_prev):
            blocks.append(i-1)#-1 because we want to only Add layers


    i = len(model.layers)-1
    while i > 0:
        if isinstance(model.get_layer(index=i), Add):
            blocks.append(i)
            break
        i = i-1

    return blocks

def fit(cnn_model, X_train, y_train,
        X_test, y_test, epochs=2):

    schedule = [(100, 1e-3), (150, 1e-4)]
    lr_scheduler = LearningRateScheduler(init_lr=0.01, schedule=schedule)
    callbacks = [lr_scheduler]

    sgd = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    cnn_model.compile(loss='categorical_crossentropy',
                      optimizer=sgd, metrics=['accuracy'])

    for ep in range(1, epochs):
        X_tmp = np.concatenate((data_augmentation(X_train),
                                data_augmentation(X_train),
                                data_augmentation(X_train)))
        y_tmp = np.concatenate((y_train,
                                y_train,
                                y_train))

        cnn_model.fit(X_tmp, y_tmp, batch_size=128,
                      callbacks=callbacks, verbose=2,
                      epochs=ep, initial_epoch=ep - 1)

        if ep % 5 == 0:
            acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(cnn_model.predict(X_test), axis=1))
            print('Accuracy [{:.4f}]'.format(acc), flush=True)

    return cnn_model

if __name__ == '__main__':
    np.random.seed(12227)

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_units', type=int, default=2)
    parser.add_argument('--n_components', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=3)

    args = parser.parse_args()
    num_units = args.num_units
    n_components = args.n_components
    iterations = args.iterations
    epochs = args.epochs

    (X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()
    X_train = X_train.astype('float32') / 255
    X_test = X_test.astype('float32') / 255
    X_train_mean = np.mean(X_train, axis=0)
    X_train -= X_train_mean
    X_test -= X_train_mean

    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    shallow_model = create_model(input_shape=(32, 32, 3), depth_block=[2, 2, 2],
                             num_classes=10, iter=-1)
    shallow_model = fit(shallow_model, X_train, y_train, X_test, y_test, epochs)

    num_res_blocks = np.array(count_res_blocks(shallow_model))

    score_shallow = score(shallow_model, X_train, y_train,
                          layers=idx_blocks(shallow_model),
                          n_components=n_components)
    K.clear_session()
    print('Score {}'.format(score_shallow), flush=True)
    allow_grow = [True, True, True]

    for i in range(0, iterations):

        tmp_model = create_model(input_shape=(32, 32, 3), depth_block=num_res_blocks+num_units,
                                      num_classes=10, iter=i)
        tmp_model = fit(tmp_model, X_train, y_train, X_test, y_test, epochs)
        score_tmp = score(tmp_model, X_train, y_train,
                          layers=idx_blocks(tmp_model),
                          n_components=n_components)
        K.clear_session()

        print('Score TMP{}'.format(score_tmp))

        allow_grow = np.where(np.array(score_tmp) > np.array(score_shallow))[0]
        allow_grow = np.in1d(np.arange(len(score_tmp)), allow_grow)

        if not any(allow_grow):
            print('#######Convergence achieved########')

        for block in range(0, len(num_res_blocks)):
            if allow_grow[block]:
                num_res_blocks[block] = num_res_blocks[block] + num_units

        deep_model = create_model(input_shape=(32, 32, 3),depth_block=num_res_blocks,
                                  num_classes=10, iter=i)
        deep_model = fit(deep_model, X_train, y_train, X_test, y_test, epochs)
        score_deep = score(deep_model, X_train, y_train,
                           layers=idx_blocks(deep_model),
                           n_components=n_components)

        print('Score {}'.format(score_deep))

        save_model('model_iteration[{}]_Units[{}]'.format(i, num_units), model=deep_model)

        y_pred = deep_model.predict(X_test)
        acc = accuracy_score(np.argmax(y_test, axis=1), np.argmax(y_pred, axis=1))

        n_params = deep_model.count_params()
        flops, _ = compute_flops(deep_model)
        print('Iteration [{}] Number of Blocks {} Number of Parameters [{}] FLOPS [{}] Accuracy [{:.4f}]'.format(i, num_res_blocks, n_params, flops, acc))
        K.clear_session()
        score_shallow = score_deep
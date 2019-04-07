import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.losses as losses
import tensorflow.keras.optimizers as optimizers
import physionet_handler


# network parameters
epochs = 500
batch_size = 100
quantity_no = 42
activation = layers.ReLU()

train_set = './data/set-a/'
label_file = './data/set-a_outcome.txt'
physionet = physionet_handler.PhysioHandler(set_path=train_set,
                                            label_file=label_file,
                                            batch_size=batch_size)
val_set = './data/set-b/'
val_label_file = './data/set-b_outcome.txt'
val_physionet = physionet_handler.PhysioHandler(set_path=val_set,
                                                label_file=val_label_file,
                                                max_length=physionet.max_length,
                                                batch_size=batch_size)
filters = [50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
kernel_size = [12, 12, 8, 6, 6, 6, 6, 6, 6, 6]
assert len(filters) == len(kernel_size)
strides = [2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
padding = 'valid'
data_format = 'channels_last'

layer_lst = []
for layer_no in range(0, len(filters)):
    layer_lst.append(layers.Conv1D(filters[layer_no], kernel_size[layer_no],
                                   strides[layer_no], padding, data_format,
                                   activation=activation))
layer_lst.append(layers.Dense(1, activation=None))

model = tf.keras.Sequential(layer_lst)
loss_fun = losses.BinaryCrossentropy(from_logits=True)
opt = optimizers.Adam(lr=0.001)
model.compile(opt, loss=loss_fun, metrics=['accuracy'])

# model.fit(x=physionet.generator(), epochs=epochs,
#           steps_per_epoch=physionet.steps_per_epoch)

val_batches = val_physionet.get_batches()
for e in range(0, epochs):
    image_batches, target_batches = physionet.get_batches()
    assert len(image_batches) == len(target_batches)
    for i in range(len(image_batches)):
        x = image_batches[i]
        y = target_batches[i]
        hist = model.fit(x=x, y=y, verbose=0)
        # print(hist.history['loss'])
    print('epoch', e+1, 'of', epochs, 'done')
    # validate
    if e % 10 == 0:
        test_acc = []
        for j in range(len(val_batches[0])):
            x_val = val_batches[0][j]
            y_val = val_batches[1][j]
            test_acc.append(model.test_on_batch(x_val, y_val, reset_metrics=True)[1])
        print('test acc', np.mean(test_acc))

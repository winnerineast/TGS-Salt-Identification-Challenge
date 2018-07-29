from keras.applications.densenet import DenseNet121
from keras import layers
from keras.layers import Input
from keras.models import Model


def MyDenseNet169(H, W, num_classes):
    input_layer = Input(shape=(H,W,3))

    dense = DenseNet121(include_top=False,
                        weights='imagenet',
                        input_shape=(H, W, 3),
                        classes=num_classes)

    x = dense(input_layer)
    x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
    x = layers.Dense(num_classes, activation='softmax', name='fc')(x)

    model = Model(inputs=input_layer, outputs=x)
    return model

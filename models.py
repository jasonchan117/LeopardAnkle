
from tensorflow.keras.layers import Conv2D,  MaxPool2D,  Flatten, Dense
from tensorflow.keras import Model

class Net(Model):
    def __init__(self, type):
        super(Net, self).__init__()
        self.c1 = Conv2D(filters=6, kernel_size=(5, 5), activation='relu',padding='same')
        self.p1 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.c2 = Conv2D(filters=16, kernel_size=(5, 5), activation='relu',padding='same')
        self.p2 = MaxPool2D(pool_size=(2, 2), strides=2)

        self.flatten = Flatten()
        self.f1 = Dense(120, activation='relu')
        self.f2 = Dense(84, activation='relu')

        if type == 'classification':
            self.f3 = Dense(31, activation='softmax')
        else:
            self.f3 = Dense(1,activation='tanh')

    def call(self, x):
        x = self.c1(x)
        x = self.p1(x)

        x = self.c2(x)
        x = self.p2(x)

        x = self.flatten(x)
        x = self.f1(x)
        x = self.f2(x)
        y = self.f3(x)
        return y
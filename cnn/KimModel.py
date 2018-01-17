from __future__ import print_function

from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Merge
from keras.layers import GlobalMaxPooling1D, ZeroPadding1D, Activation

P_DROPOUT = 0.5
MINI_BATCH_SIZE = 50
LAMBDA = 3
EPOCHS = 150

class KimModel:
    def __init__(self, wordVectSize, numOfWords=25, numOfFilters=100):
        self.__numberOfWords = numOfWords
        self.__sizeOfWordVectors = wordVectSize
        self.__numberOfFilter = numOfFilters
        self.__model = self.__createModel()

    def __createModel(self):
        # [NOTED] Kim used 2-channels input, static and non-static channels (Section 2, Page 2)
        model = Sequential()
        multiFilterSizeConvolution = []

        for h in [3, 4, 5]:
            submodel = Sequential()
            submodel.add(ZeroPadding1D(
                padding=1, 
                input_shape=(self.__numberOfWords, self.__sizeOfWordVectors)))
            submodel.add(Conv1D(
                    filters=self.__numberOfFilter, 
                    kernel_size=h, 
                    padding='valid', 
                    activation='relu', 
                    strides=1,
                    kernel_regularizer=regularizers.l2(LAMBDA)))
            submodel.add(GlobalMaxPooling1D())

            multiFilterSizeConvolution.append(submodel)

        #  UserWarning: The `Merge` layer is deprecated and will be removed after 08/2017. 
        #  Use instead layers from `keras.layers.merge`, e.g. `add`, `concatenate`, etc.
        model.add(Merge(multiFilterSizeConvolution, mode="concat"))
        model.add(Dropout(P_DROPOUT))
        model.add(Activation('softmax'))
        print('Compiling model')
        model.compile(loss='binary_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])
        return model

    def train(self, X, Y):
        print("Start training")
        self.__model.fit([X, X, X], Y, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE)
        print("Training complete")

    def save(self):
        pass

    def predict(self):
        pass





from __future__ import print_function

from keras import regularizers
from keras.models import Sequential
from keras.layers import Conv1D, Dropout, Merge, Dense
from keras.layers import GlobalMaxPooling1D, ZeroPadding1D, Activation
from keras.models import model_from_json

P_DROPOUT = 0.5
MINI_BATCH_SIZE = 50
LAMBDA = 3
EPOCHS = 5

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
        model.add(Dense(1, input_shape=(300,)))
        model.add(Activation('softmax'))
        print('Compiling model')
        model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['accuracy'])
        return model

    def train(self, X, Y):
        print("Start training")
        self.__model.fit([X, X, X], Y, epochs=EPOCHS, batch_size=MINI_BATCH_SIZE)
        print("Training completed")

    def test(self, X, Y):
        print("Start testing")
        scores = self.__model.evaluate([X, X, X], Y, batch_size=MINI_BATCH_SIZE)
        print("Testing completed")
        return scores

    def save(self, modelName="model.json", weightName="weight.h5"):
        modelJson = self.__model.to_json()
        with open(modelName, "w") as jsonFile:
            jsonFile.write(modelJson)
        self.__model.save_weights(weightName)

    def load(self, modelName="model.json", weightName="weight.h5"):
        jsonFile = open(modelName, 'r')
        modelJson = jsonFile.read()
        jsonFile.close()
        self.__model = model_from_json(modelJson)
        self.__model.compile(
            loss='mean_squared_error',
            optimizer='adadelta',
            metrics=['accuracy'])
        self.__model.load_weights(weightName)

    def predict(self, sentence):
        return self.__model.predict([sentence, sentence, sentence])
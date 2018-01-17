from __future__ import print_function

import os
import word2vec
import numpy

class W2VProcessor:
    def __init__(self, originData=None, dataFolder="", vectorSize=100):
        self.__model = None
        self.__vectorSize = vectorSize
        if type(originData) is str:
            # word2vec.word2phrase(
            #     originData, 
            #     os.path.join(dataFolder, "phrases.phr"), 
            #     verbose=True)
            # originData = os.path.join(dataFolder, "phrases.phr")
            word2vec.word2vec(
                originData, 
                os.path.join(dataFolder, "vec.w2v"), 
                size=vectorSize, 
                verbose=True)
            self.__model = word2vec.load(os.path.join(dataFolder, "vec.w2v"))

    def load(self, wordVectorFile):
        self.__model = word2vec.load(wordVectorFile)
        self.__vectorSize = self.__model.vectors.shape[1]

    def getVectorSize(self):
        return self.__vectorSize

    def process(self, sentence, length=None):
        if self.__model is None:
            print("Error: The model is None")
            return None

        if not sentence:
            print("Error: The sentence is None")
            return None

        if length is None:
            length = len(sentence)
        sentence = sentence[:length]
        tensor = []
        for word in sentence:
            try:
                tensor.append(self.__model[word])
            except:
                tensor.append(numpy.zeros((self.__vectorSize,)))
        for i in range(length-len(sentence)):
            tensor.append(numpy.zeros(tensor[0].shape))
        tensor = numpy.concatenate(tensor).reshape((length, len(tensor[0])))

        return tensor





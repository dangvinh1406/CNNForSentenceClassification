from __future__ import print_function

import os
import word2vec
import numpy

class W2VProcessor:
    def __init__(self, originData=None, dataFolder="", vectorSize=100):
        self.__model = None
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

    def process(self, sentence):
        if self.__model is None:
            print("Error: The model is None")
            return None
        tensor = []
        for word in sentence:
            tensor.append(self.__model[word])
        tensor = numpy.concatenate(tensor).reshape(
            (len(sentence), len(tensor[0])))
        return tensor





from __future__ import print_function

import os
import sys
import shutil
import numpy
import time
import glob
import random

from cnn.KimModel import KimModel
from cnn.W2VProcessor import W2VProcessor
from cnn.Preprocessor import Tokenizer, Filter, Utilities

def convertDataToTensors(
    extractor, sentenceSize, 
    dataFileName, outputFolder, label):
    stringContent = ""
    with open(dataFileName, "r") as file:
        stringContent = file.read()
    sentences = Tokenizer.tokenizeSentence(stringContent)
    i = 0
    prefix = str(int(time.time()))
    for sentence in sentences:
        words = Tokenizer.tokenizeWord(sentence)
        tensor = extractor.process(words, sentenceSize)
        if tensor is None:
            continue
        ouputName = os.path.join(
            outputFolder, label+"_"+prefix+str(i))+".npy"
        numpy.save(ouputName, tensor)
        print("\rSaved tensor "+ouputName+" size "+str(tensor.shape), end="")
        i += 1

def collectTensors(tensorFolder, ratio=0.7):
    tensorFiles = glob.glob(os.path.join(tensorFolder, "*.npy"))
    random.shuffle(tensorFiles)
    X = []
    Y = []
    for filename in tensorFiles:
        X.append(numpy.load(filename))
        Y.append(int(filename.split("/")[-1][0]))
        print("\rLoaded tensor "+filename, end="")
    print("\nData contains "+str(len(X))+" records ")
    XTrain = numpy.array(X[:int(ratio*len(Y))])
    YTrain = Y[:int(ratio*len(Y))]
    XTest = numpy.array(X[int(ratio*len(Y)):])
    YTest = Y[int(ratio*len(Y)):]
    print("Training data: ", XTrain.shape)
    print("Testing data: ", XTest.shape)
    return XTrain, YTrain, XTest, YTest


if __name__ == '__main__':
    sentenceSize = 50

    dataFolder = "/home/sidratul/repositories/CNNForSentenceClassification/data/rt-polaritydata/"
    outputFolder = "/home/sidratul/repositories/CNNForSentenceClassification/data/tensors/"
    w2vTrainedModel = "/home/sidratul/repositories/word2vec/vectors.bin"
    modelFile = "/home/sidratul/repositories/CNNForSentenceClassification/data/model.h5"

    shutil.rmtree(outputFolder)
    os.mkdir(outputFolder)

    extractor = W2VProcessor()
    extractor.load(w2vTrainedModel)

    model = KimModel(
        wordVectSize=extractor.getVectorSize(), 
        numOfWords=sentenceSize, 
        numOfFilters=100)

    convertDataToTensors(
        extractor, sentenceSize, 
        dataFolder+"rt-polarity.neg", outputFolder, "0")
    convertDataToTensors(
        extractor, sentenceSize, 
        dataFolder+"rt-polarity.pos", outputFolder, "1")

    X, Y, x, y = collectTensors(outputFolder)
    model.train(X, Y)
    model.save(modelFile)
    model.load(modelFile)
    scores = model.test(x, y)
    print("Accuracy: ", scores)
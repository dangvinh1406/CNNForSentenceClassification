import re
import itertools

class Tokenizer:
    @staticmethod
    def tokenizeSentence(raw):
        """
        Function tokenizes a string to sentences based the character "new line"
        """
        if type(raw) is not str:
            return []
        return raw.split("\n")

    @staticmethod
    def tokenizeWord(raw):
        """
        Function tokenizes a string to words based the non-word characters
        """
        if type(raw) is not str:
            return []
        return re.findall(r"[\w]+", raw)

class Filter:
    REPEAT_REGEXP = re.compile(r'(\w*)(\w)\2(\w*)')
    REPL = r'\1\2\3'

    @staticmethod
    def filterWord(listOfWords, blackSet):
        """
        Function filters out all stop words and numbers
        """
        return [word for word in listOfWords 
            if word not in blackSet 
            and not word.isdigit()]


    @staticmethod
    def filterSentence(listOfSentences, numberOfWordsPerSentence):
        """
        Function filters out all sentences which have less than a number of words
        """
        return [l for l in listOfSentences if len(l) > numberOfWordsPerSentence]

    @staticmethod
    def filterCharacter(word, dictionary):
        """
        Function filters all repeated characters of a word
        """
        if word in dictionary:
            return word
        replWord = Filter.REPEAT_REGEXP.sub(Filter.REPL, word)
        if replWord != word:
            return Filter.filterCharacter(replWord, dictionary)
        else:
            return replWord


class Utilities:

    @staticmethod
    def convertLowercase(string):
        return string.lower()

    @staticmethod
    def flattenListOfList(listOfList):
        return list(itertools.chain.from_iterable(listOfList))
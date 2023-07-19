from nltk import RegexpTokenizer
import nltk
import os
import json
import re
import pickle
import glob
from sklearn.model_selection import train_test_split


def ReadJson(path):
    """
    Reading Json files. It needs "Jsons" folder in current directory. It can take a Turkish characters with
    encoding="utf-8"
    """
    folder_path = path
    temp = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                targetText = data["ictihat"]
                targetText = targetText.lower()
                temp.append(targetText)

    return temp


def SplitTestAndTrain():
    try:
        json_files = glob.glob(os.path.join(os.getcwd() + "\\Jsons", '*.json'))

        train_files, test_files = train_test_split(json_files, test_size=0.3, random_state=42)

        if not os.path.exists("Jsons/Train"):
            os.makedirs("Jsons/Train")
        for file in train_files:
            os.rename(file, "Jsons/Train/" + os.path.basename(file))

        if not os.path.exists("Jsons/Test"):
            os.makedirs("Jsons/Test")
        for file in test_files:
            os.rename(file, "Jsons/Test/" + os.path.basename(file))
    except:
        pass


def SaveAsPKL(ngram_models):
    with open('nGramModel.pkl', 'wb') as f:
        pickle.dump(ngram_models, f)


def GetPKL():
    with open('nGramModel.pkl', "rb") as file:
        model = pickle.load(file)
        return model


def TrainSet(n, ngram_models, stopWords, text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    words = tokenizer.tokenize(text.lower())
    filtered_tokens = [token for token in words if token.lower() not in stopWords]

    ngrams = nltk.ngrams(filtered_tokens, int(n))
    freq_dist = nltk.FreqDist(ngrams)
    total = freq_dist.N()
    for ngram in freq_dist:
        freq = freq_dist[ngram]
        prob = freq / total
        freq_dist[ngram] = prob
    ngram_models[str(n) + "gram"] = freq_dist
    return ngram_models


def NGramPhase(n, text):
    # text = Json()
    words = []
    for word in re.split(r'[^\w\s]+|\s', text.lower()):  # Delete all special characters.
        if not word.isnumeric():  # Delete also numeric elements.
            words.append(word)

    text = ''.join(words).lower()  # List to string.

    pieces = [text[i:i + n] for i in range(len(text) - n + 1)]
    freq_dist = nltk.FreqDist(pieces)
    return freq_dist


def NGramCompleteWord(targetWord, text):
    tokenizer = RegexpTokenizer(r'\w+')  # Get tokens without special characters.
    words = tokenizer.tokenize(text.lower())  # Get words list with lowercase letters.
    targetWord = targetWord.lower()
    frequencyDict = {}
    for word in words:
        if word[0: len(targetWord)] == targetWord[0: len(word)]:
            if word.__contains__(targetWord):  # If target word is in the files it adds to dictionary.
                if word in frequencyDict.keys():  # If there is the target word in the dictionary, counts
                    frequencyDict[word] += 1  # will one more.
                else:
                    frequencyDict[word] = 1

    frequencyDict = list(sorted(frequencyDict.items(), key=lambda x: x[1], reverse=True))
    if len(frequencyDict) == 0:
        return "There is no matching word!"
    return frequencyDict[0][0].capitalize()  # Sort and return most popular word.


def GetTrainedDataSet(ngramModel, N_WordCount, key):
    newDict = {}
    temp = 0.0
    if key is None:                                                                 # for characters
        newDict = ngramModel[(str(N_WordCount) + 'gramChar')]
        newDictSorted = dict(sorted(newDict.items(), key=lambda x: x[1], reverse=True))
        return newDictSorted
    else:
        for DictKey, DictValue in ngramModel[str(N_WordCount) + 'gram'].items():
            if DictKey[0] == key.lower():                                                   # for words
                temp = temp + DictValue
                newDict[DictKey] = DictValue
        if temp == 0:
            print("The word cannot scripted!")
            exit()

        multiplier = 100 / temp
        for DictKey, DictValue in newDict.items():
            newDict[DictKey] = DictValue * multiplier
        newDictSorted = dict(sorted(newDict.items(), key=lambda x: x[1], reverse=True))
        return newDictSorted


def Main():
    try:
        SplitTestAndTrain()
    except:
        pass

    testContent = ReadJson(os.getcwd() + "/Jsons/Test")
    testText = ' '.join(testContent)                                # Translation to string

    trainContent = ReadJson(os.getcwd() + "/Jsons/Train")
    trainText = ' '.join(trainContent)

    ngramWordModel = {}

    try:
        ngramWordModel = GetPKL()
    except:
        pass

    stopwords = set(nltk.corpus.stopwords.words('turkish'))  # Get some stopwords from nltk.corpus
    stopwordList = []
    with open("stopword.txt", "r", encoding="UTF-8") as file:  # Reading stopword.txt.
        content = file.read()
        word = content.split("\n")
        for i in range(len(word) - 1):
            stopwordList.append(word[i])

    stopwords.update(stopwordList)

    while True:
        choice = input("1. N-Gram Word\n2. Most Common Character Sets\n3. Complete to the most frequently repeated word"
                       "\n4. Complete Next Word\n5. Exit\nPlease Choose Operation: ")

        if choice == '1':
            N_WordCount = input("Please Enter N-Word Count: ")
            key = input("Please Enter the Key: ")

            if ngramWordModel.keys().__contains__(str(N_WordCount) + 'gram'):
                print("This model is trained for this n:", N_WordCount)
            else:
                print("Model is being trained for this n:", N_WordCount)
                model = TrainSet(N_WordCount, ngramWordModel, stopwords, trainText)
                SaveAsPKL(model)

            model = GetTrainedDataSet(ngramWordModel, N_WordCount, key)
            print("N Gram Word: ", model)

        elif choice == '2':
            N_WordCount = int(input("Please Enter N-Word Count: "))
            if ngramWordModel.keys().__contains__(str(N_WordCount) + 'gramChar'):
                print("This model is trained for this", N_WordCount)
            else:
                print("Model is being trained for this n:", N_WordCount)
                model = NGramPhase(N_WordCount, trainText)
                ngramWordModel[str(N_WordCount) + "gramChar"] = model
                SaveAsPKL(ngramWordModel)

            model = GetTrainedDataSet(ngramWordModel, N_WordCount, None)
            print("N Gram Phase: ", model)

        elif choice == '3':
            targetWord = input("Please enter character sets: ")
            calculatedWord = NGramCompleteWord(targetWord, trainText)
            print("Completed word:", calculatedWord)

        elif choice == '4':
            key = input("Please Enter the Key: ")
            if ngramWordModel.keys().__contains__(str(2) + 'gram'):
                print("This model is trained for this n:", 2)

            else:
                print("Model is being trained for this n:", 2)
                model = TrainSet(2, ngramWordModel, stopwords, trainText)
                SaveAsPKL(model)

            model = GetTrainedDataSet(ngramWordModel, 2, key)
            predict = list(model.keys())[-1][1]
            print("Most Popular Next Word: ", predict.capitalize(), "\n")
        elif choice == '5':
            exit()
        else:
            print("\nInvalid Input!\n")
            continue


Main()

# perplexity = exp(-1 * sum(log2(model(test_set[i-1:i])) for i in range(1, len(test_set)+1)) / len(test_set)

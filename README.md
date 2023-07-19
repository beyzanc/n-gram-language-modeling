# N-Gram Language Modeling

This repository contains an implementation of an N-Gram language model for word and character prediction. The model is built using the NLTK library in Python and can be used to generate predictions based on a given input text.

## The approach to the problem

The given dataset was first edited to create a word and character-based n-gram language model. Unnecessary characters, punctuation, numbers, and other such elements were removed from the dataset. A stopword.txt file was created, and commonly used irrelevant words were added to it during its creation.

![](1)

The word-based n-gram model was built by individually separating all the words in the dataset and then splitting them into n-grams. For example, to create the 2-gram model, a pair was created for every three consecutive words.

Similarly, for the character-based n-gram model, each character in the dataset was split into n-grams. For instance, the 3-gram model was generated by creating a triplet for every four consecutive characters.

To calculate the word- or character-based n-gram models, the number of n-gram pairs or triplets generated was determined. The frequency of the created n-grams was calculated to find out how many times each n-gram appeared in the dataset.

Subsequently, the word or words with the highest frequency were calculated to predict the word that comes after a specific word. The graphs below illustrate the top 10 most frequently used words following "hukuk" and "kasten."

![](2)
![](3)
![](4)
![](5)

## Library

- **import nltk**
nltk can be used for analysis, classification, summarization, translation, phrase separation, rooting, calculating the frequency of words and much more.
In this code, the word_tokenize() function of the nltk library is used to split a text document into words. This function performs phrase separation and returns the words in the text as a list.

-	**from nltk import RegexpTokenizer**
The RegexpTokenizer class uses regular expressions to segment or split the text into chunks. Splits a text into pieces according to the specified regular expression and names each piece a "token".
In this code, the regular expression RegexpTokenizer(r'\w+') refers to one or more occurrences of any word character. That is, the tokenizer object splits a text string according to this regular expression, returning a list representing each word as a separate "token".

- **import os**
It is used the os library's listdir() and join() functions to manipulate files in a folder.

-	**import json**
It is used the Python programming language json library's load() function to read data from a JSON file.

-	**import glob**
The glob library is a Python module for listing the names of files or directories in a folder. The glob.glob function returns all filenames that match a given filename pattern. In the code, by adding all the files matching the "Json" pattern into the json_files list, it can be processed these files one by one in subsequent operations.

- **from sklearn.model_selection import train_test_split**
train_test_split is a function in sklearn.model_selection module of the scikit-learn library that is used to split a dataset into training and testing sets for the purpose of machine learning.
In this code, train_test_split is used to split the json_files list into two separate lists: *train_files* and *test_files*. The test_size parameter is set to *0.3*, which means that **30% of the data will be used for testing and 70% will be used for training**. The random_state parameter is set to 42 to ensure that the split is the same every time the code is run, which is important for reproducibility of results.

- **import pickle**
Pickle is used to save the ngram_models dictionary as a binary file with the extension .pkl. This allows the model to be saved and loaded later without having to retrain the model every time.

## The Results & Methods

It have prepared a menu for the n-gram language model.

![](6)

### 1. N-Gram Word

#### def NGramWord(text, n, stopWords):

The purpose of this function is to calculate the frequency of N-Gram words in a given text string. In this way, the word or phrase with the highest frequency becomes the word or phrase that wanted to predict. It checks whether the dataset is trained or not and trains the model accordingly.

The function converts the text to lowercase and tokenizes it using the RegexpTokenizer class from the nltk.tokenize module. It then removes any words that appear in the stopWords list. The function returns the ngramsDictionary, which contains the N-Gram words as keys and their frequency as values. The frequency values represent the percentage of usage of each N-Gram word in the text.

To use this function, it should be called it with a text string, an integer value n indicating the desired N-Gram size (e.g. 2 for bigrams, 3 for trigrams), and a list of stopWords that should be ignored in the N-Gram calculations. The function would then return a dictionary containing the N-Gram words and their frequencies in the input text.

### 2. Most Common Character Sets

#### def NGramPhase(text, n):

The purpose of this function is to determine the frequency of occurrence of characters as much as the number it was received as input. It can be used the most common character to predict the continuation of the word. The same process was applied to the characters as was done for the words in the previous function.

The function eventually returns a dictionary of N-Gram characters in the input text and their frequencies, sorted by frequency in ascending order. It was added the part of it because the output is too long.

![](7)
![](8)

### 3. Complete to the most frequently repeated Word

##### def NGramCompleteWord(text, targetWord):

This function aims to find the word in the input text that starts with the same characters as the targetWord parameter and has the highest frequency. This way function can predict how the word will continue.

![](9)

### 4. Complete Next Word

#### def NGramWord(text, n, stopWords):

It was used the function that used in “1 .N-Gram Word” here as well. It was used the word with the highest frequency to predict the next word after the word it was received with the input.

![](10)

### 5. Train set

#### def TrainSet(text, n, ngram_models, stopWords):

The function first converts the text to lowercase and tokenizes it by using RegexpTokenizer from the NLTK library. It then filters out any stop words from the list of tokens. Next, it uses the nltk.ngrams function to generate n-grams of length n from the filtered tokens. The frequency distribution of these n-grams is calculated using the nltk.FreqDist function. For each n-gram in the frequency distribution, the function calculates its probability by dividing its frequency by the total number of n-grams. The frequency distribution is then updated with the probabilities. The function returns the updated dictionary of n-gram models.


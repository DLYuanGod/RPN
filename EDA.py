import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import numpy as np
import random
import nltk
from nltk.corpus import wordnet
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])


def eda_SR(originalSentence, n):
    """
    Paper Methodology -> Randomly choose n words from the sentence that are not stop words.
                        Replace each of these words with one of its synonyms chosen at random.
    originalSentence -> The sentence on which EDA is to be applied
    n -> The number of words to be chosen for random synonym replacement
    """
    stops = set(stopwords.words('english'))
    splitSentence = list(originalSentence.split(" "))
    splitSentenceCopy = splitSentence.copy()
    # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
    ls_nonStopWordIndexes = []
    for i in range(len(splitSentence)):
        if splitSentence[i].lower() not in stops:
            ls_nonStopWordIndexes.append(i)
    if (n > len(ls_nonStopWordIndexes)):
        raise Exception(
            "The number of replacements exceeds the number of non stop word words")
    for i in range(n):
        indexChosen = random.choice(ls_nonStopWordIndexes)
        ls_nonStopWordIndexes.remove(indexChosen)
        synonyms = []
        originalWord = splitSentenceCopy[indexChosen]
        for synset in wordnet.synsets(originalWord):
            for lemma in synset.lemmas():
                if lemma.name() != originalWord:
                    synonyms.append(lemma.name())
        if (synonyms == []):
            continue
        splitSentence[indexChosen] = random.choice(synonyms).replace('_', ' ')
    return " ".join(splitSentence)


def eda_RI(originalSentence, n):
    """
    Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word.
                        Insert that synonym into a random position in the sentence. Do this n times
    originalSentence -> The sentence on which EDA is to be applied
    n -> The number of times the process has to be repeated
    """
    stops = set(stopwords.words('english'))
    splitSentence = list(originalSentence.split(" "))
    splitSentenceCopy = splitSentence.copy() 
    # Since We Make Changes to The Original Sentence List The Indexes Change and Hence an initial copy proves useful to get values
    ls_nonStopWordIndexes = []
    for i in range(len(splitSentence)):
        if splitSentence[i].lower() not in stops:
            ls_nonStopWordIndexes.append(i)
    if (n > len(ls_nonStopWordIndexes)):
        raise Exception("The number of replacements exceeds the number of non stop word words")
    WordCount = len(splitSentence)
    for i in range(n):
        indexChosen = random.choice(ls_nonStopWordIndexes)
        ls_nonStopWordIndexes.remove(indexChosen)
        synonyms = []
        originalWord = splitSentenceCopy[indexChosen]
        for synset in wordnet.synsets(originalWord):
            for lemma in synset.lemmas():
                if lemma.name() != originalWord:
                    synonyms.append(lemma.name())
        if (synonyms == []):
            continue
        splitSentence.insert(random.randint(0,WordCount-1), random.choice(synonyms).replace('_', ' '))
    return " ".join(splitSentence)

def eda_RS(originalSentence, n):
    """
    Paper Methodology -> Find a random synonym of a random word in the sentence that is not a stop word. 
                        Insert that synonym into a random position in the sentence. Do this n times
    originalSentence -> The sentence on which EDA is to be applied
    n -> The number of times the process has to be repeated
  """
    splitSentence = list(originalSentence.split(" "))
    WordCount = len(splitSentence)
    for i in range(n):
        firstIndex = random.randint(0,WordCount-1)
        secondIndex = random.randint(0,WordCount-1)
        while (secondIndex == firstIndex and WordCount != 1):
            secondIndex = random.randint(0,WordCount-1)
    splitSentence[firstIndex], splitSentence[secondIndex] = splitSentence[secondIndex], splitSentence[firstIndex]
    return " ".join(splitSentence)

def eda_RD(originalSentence, p):
    """
    Paper Methodology -> Randomly remove each word in the sentence with probability p.
    originalSentence -> The sentence on which EDA is to be applied
    p -> Probability of a Word Being Removed
    """
    og = originalSentence
    if (p == 1):
        raise Exception("Always an Empty String Will Be Returned") 
    if (p > 1 or p < 0):
        raise Exception("Improper Probability Value")
    splitSentence = list(originalSentence.split(" "))
    lsIndexesRemoved = []
    WordCount = len(splitSentence)
    for i in range(WordCount):
        randomDraw = random.random()
        if randomDraw <= p:
            lsIndexesRemoved.append(i)
    lsRetainingWords = []
    for i in range(len(splitSentence)):
        if i not in lsIndexesRemoved:
            lsRetainingWords.append(splitSentence[i])
    if (lsRetainingWords == []):
        return og
    return " ".join(lsRetainingWords)

# EDA函数
def eda(sentence, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=9):

    words = list(sentence.split())
    num_words = len(words)

    augmented_sentences = []
    num_new_per_technique = int(num_aug / 4) + 1
    n_sr = max(1, int(alpha_sr * num_words))
    n_ri = max(1, int(alpha_ri * num_words))
    n_rs = max(1, int(alpha_rs * num_words))

    # print(words, "\n")

    # 同义词替换sr
    for _ in range(num_new_per_technique):
        a_sentence = eda_SR(sentence, n_sr)
        augmented_sentences.append(a_sentence)

    # 随机插入ri
    for _ in range(num_new_per_technique):
        a_sentence = eda_RI(sentence, n_ri)
        augmented_sentences.append(a_sentence)

    # 随机交换rs
    for _ in range(num_new_per_technique):
        a_sentence = eda_RS(sentence, n_rs)
        augmented_sentences.append(a_sentence)

    # 随机删除rd
    for _ in range(num_new_per_technique):
        a_sentence = eda_RD(sentence, p_rd)
        augmented_sentences.append(a_sentence)

    # print('*'*20, len(augmented_sentences))
    shuffle(augmented_sentences)

    if num_aug >= 1:
        augmented_sentences = augmented_sentences[:num_aug]
    else:
        keep_prob = num_aug / len(augmented_sentences)
        augmented_sentences = [s for s in augmented_sentences if random.uniform(0, 1) < keep_prob]


    return augmented_sentences

def get_eda_df(sentences, alpha=0.1, num_avg=1):
    results = []
    for i, sents in enumerate(sentences):
        augmented_sentences = eda(sents, alpha_sr=alpha, alpha_ri=alpha, alpha_rs=alpha, p_rd=alpha,
                                  num_aug=num_avg)
        results.append(" ".join(augmented_sentences))
    return results


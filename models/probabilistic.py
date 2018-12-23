#!/usr/bin/env python3

import math
import numpy as np
import string


class Probabilistic:
    """
    Calculates required probabilities for prediction
    """

    def fit(self, X):
        words = self._clean(X)
        self._build_dictionary(words)

        words = self._index(words)
        self._calculate_transition_1_prob(words)
        self._calculate_transition_2_prob(words)

    """
    Generates new word using 1 or 2 seed words
    """

    def next(self, word_i_1, word_i_2=None):
        if word_i_2 is None:
            word_i_1 = self.word_index[word_i_1]
            candidates = list(self.transition_1_prob[word_i_1].keys())
            distribution = list(self.transition_1_prob[word_i_1].values())
        else:
            word_i_1 = self.word_index[word_i_1]
            word_i_2 = self.word_index[word_i_2]
            candidates = list(self.transition_2_prob[word_i_2][word_i_1].keys())
            distribution = list(self.transition_2_prob[word_i_2][word_i_1].values())

        choice = np.random.choice(candidates, size=1, p=distribution)[0]
        return self.word_value[choice]

    # end of methods to override

    def __init__(self):
        """
        Total number of unique words in training text
        """
        self.num_words = 0

        """
        Stores numerical index for each word
        e.g. "apple": 0, "cat": 1, ...
        """
        self.word_index = None

        """
        Stores value of word stored at index
        e.g. 0: "apple", 1: "cat", ...
        """
        self.word_value = None

        """
        Transition 1 cost is the negative log of probability of having some word after some other word.
        E.g. probablity of 'apple' being followed by 'cat', P(word_i|word_i-1)

        Structure:
        [
            word_i=0: [
                word_i-1=0, word_i-1=1, ...
            ],
            ...
        ]
        """
        self.transition_1_prob = None

        """
        Transition 2 cost is the negative log of probability of having some word given values of 2 previous words.
        E.g. probablity of apple -> ate -> cat, P(word_i|word_i-1, word_i-2)

        Structure:
        [
            word_i=0: [
                word_i-1=0: [
                    word_i-2=0, word_i-2=1, ...
                ],
                word_i-1=1, ...
            ],
            ...
        ]
        """
        self.transition_2_prob = None

        # Hyperparameters
        self.MISSING_WORD_PROBABILITY = 10e-5
        self.MISSING_WORD_COST = -np.log(self.MISSING_WORD_PROBABILITY)
        self.MISSING_transition_1_prob = -np.log(10e-12)
        self.MISSING_transition_2_prob = -np.log(10e-12)

    # Wrapper functions to handle errors and missing values

    """
    Input
    word_i: index of word
    word_i_1: index of previous word

    Output
    transition cost word_i_1 -> word_i: P(word_i|word_i_1)
    """

    def get_transition_1_prob(self, word_i, word_i_1):
        if word_i not in self.transition_1_prob:
            return self.MISSING_transition_1_prob
        if word_i_1 not in self.transition_1_prob[word_i]:
            return self.MISSING_transition_1_prob

        return self.transition_1_prob[word_i][word_i_1]

    """
    Input
    word_i: index of word
    word_i_1: index of previous word
    word_i_2: index of previous to previous word

    Output
    transition cost word_i_2 -> word_i_1 -> word_i: P(word_i|word_i_1, word_i_2)
    """

    def get_transition_2_prob(self, word_i, word_i_1, word_i_2):
        if word_i not in self.get_transition_2_prob:
            return self.MISSING_transition_2_prob
        if word_i_1 not in self.transition_2_prob[word_i]:
            return self.MISSING_transition_2_prob
        if word_i_2 not in self.transition_2_prob[word_i][word_i_1]:
            return self.MISSING_transition_2_prob

        return self.transition_2_prob[word_i][word_i_1][word_i_2]
    # End of Wrapper functions to handle errors and missing values

    """
    Preprocessing to allow easier learning

    Returns cleaned training words split by space
    """

    def _clean(self, text):
        # Remove punctuation
        text = ''.join(ch for ch in text if ch not in string.punctuation)

        # Lowercase
        text = text.lower()

        # Remove invalid characters
        text = text.encode('utf8').decode('ascii', 'ignore')

        return text.split()

    """
    Finds all the unique words from given text and generates 2 dictionaries: word->index, index->word

    Input
    text: training words
    """

    def _build_dictionary(self, words):
        # Get all unique words from training dataset and index both ways
        unique_words = set(words)
        self.word_index = {word: idx for (idx, word) in enumerate(unique_words)}
        self.word_index['UNK'] = len(self.word_index)
        self.word_value = {idx: word for (word, idx) in self.word_index.items()}
        self.num_words = len(self.word_index)

    def _index(self, words):
        return [self.word_index[word] if word in self.word_index else self.word_index['UNK'] for word in words]

    """
    Calculates transition probability: P(word_i|word_i-1)

    Input
    words: training words
    """

    def _calculate_transition_1_prob(self, words):
        self.transition_1_prob = {}
        for idx in range(1, len(words)):
            if words[idx - 1] not in self.transition_1_prob:
                self.transition_1_prob[words[idx - 1]] = {}
            if words[idx] not in self.transition_1_prob[words[idx - 1]]:
                self.transition_1_prob[words[idx - 1]][words[idx]] = 0
            self.transition_1_prob[words[idx - 1]][words[idx]] += 1

        # divide by sum to get probabilities
        for word_i_1 in self.transition_1_prob:
            total = sum(self.transition_1_prob[word_i_1].values())
            for word_i in self.transition_1_prob[word_i_1]:
                if total == 0:
                    self.transition_1_prob[word_i_1][word_i] = self.MISSING_WORD_PROBABILITY
                else:
                    self.transition_1_prob[word_i_1][word_i] /= total
    """
    Calculates transition probability: P(word_i|word_i-1,word_i-2)

    Input
    words: training words
    """

    def _calculate_transition_2_prob(self, words):
        self.transition_2_prob = {}
        for idx in range(2, len(words)):
            if words[idx - 2] not in self.transition_2_prob:
                self.transition_2_prob[words[idx - 2]] = {}
            if words[idx - 1] not in self.transition_2_prob[words[idx - 2]]:
                self.transition_2_prob[words[idx - 2]][words[idx - 1]] = {}
            if words[idx] not in self.transition_2_prob[words[idx - 2]][words[idx - 1]]:
                self.transition_2_prob[words[idx - 2]][words[idx - 1]][words[idx]] = 0
            self.transition_2_prob[words[idx - 2]][words[idx - 1]][words[idx]] += 1

        # divide by sum to get probabilities
        for word_i_2 in self.transition_2_prob:
            for word_i_1 in self.transition_2_prob[word_i_2]:
                total = sum(self.transition_2_prob[word_i_2][word_i_1].values())
                for word_i in self.transition_2_prob[word_i_2][word_i_1]:
                    if total == 0:
                        self.transition_2_prob[word_i_2][word_i_1][word_i] = self.MISSING_WORD_PROBABILITY
                    else:
                        self.transition_2_prob[word_i_2][word_i_1][word_i] /= total

import pandas as pd
import numpy as np
import sys

class NaiveBayesFilter:
    def __init__(self):
        self.data = []
        self.vocabulary = []  # returns tuple of unique words
        self.p_spam = 0  # Probability of Spam
        self.p_ham = 0  # Probability of Ham
        # Initiate parameters
        self.parameters_spam = {unique_word: 0 for unique_word in self.vocabulary}
        print('parameters_spam: ', self.parameters_spam)
        self.parameters_ham = {unique_word: 0 for unique_word in self.vocabulary}
        print('parameters_ham: ', self.parameters_spam)

    def fit(self, X, y):
        # Count the number of occurrences of each word in each message.
        word_counts = X.apply(lambda x: x.split())
        word_counts = word_counts.explode('word')
        word_counts = word_counts.groupby('word').size()

        # Create a DataFrame with a column for each unique word and a column for the label.
        self.data = pd.DataFrame({'word': word_counts.index, 'count': word_counts.values, 'label': y})

        # Calculate the probability of spam and ham.
        self.p_spam = self.data[self.data['label'] == 'spam'].shape[0] / self.data.shape[0]
        self.p_ham = self.data[self.data['label'] == 'ham'].shape[0] / self.data.shape[0]

        # Calculate the parameters for each word.
        for word in self.data['word'].unique():
            self.parameters_spam[word] = self.data[(self.data['word'] == word) & (self.data['label'] == 'spam')].shape[0] / self.data[self.data['word'] == word].shape[0]
            self.parameters_ham[word] = self.data[(self.data['word'] == word) & (self.data['label'] == 'ham')].shape[0] / self.data[self.data['word'] == word].shape[0]
        return self.data

    def predict_proba(self, X):
        """
        Predict the probability of each message being spam or ham.

        Args:
            X: A list of messages.

        Returns:
            A list of tuples, where each tuple contains the probability of the message being spam and the probability of the message being ham.
        """

        prob = []
        for message in X:
            # Count the number of occurrences of each word in the message.
            word_counts = message.split()
            word_counts = pd.DataFrame({'word': word_counts})

            # Calculate the probability of the message being spam and ham.
            p_spam = self.p_spam * np.prod([self.parameters_spam[word] for word in word_counts['word']])
            p_ham = self.p_ham * np.prod([self.parameters_ham[word] for word in word_counts['word']])

            prob.append((p_spam, p_ham))

        return prob


    def predict(self, X):
        """
        Predict the class of each message.

        Args:
            X: A list of messages.

        Returns:
            A list of labels, where each label is either 'spam' or 'ham'.
        """

        predictions = []
        for message in X:
            # Count the number of occurrences of each word in the message.
            word_counts = message.split()
            word_counts = pd.DataFrame({'word': word_counts})

            # Calculate the probability of the message being spam and ham.
            p_spam = self.p_spam * np.prod([self.parameters_spam[word] for word in word_counts['word']])
            p_ham = self.p_ham * np.prod([self.parameters_ham[word] for word in word_counts['word']])

            # Predict the class of the message.
            if p_spam > p_ham:
                predictions.append('spam')
            else:
                predictions.append('ham')

        return predictions


    def score(self, true_labels, predict_labels):
        recall = 0
        for i in range(len(true_labels)):
            if true_labels[i] == predict_labels[i]:
                recall += 1
            recall /= len(true_labels)
        return recall


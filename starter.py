import pandas
import numpy
from collections import defaultdict
# Read the dataset
emails = pandas.read_csv("emails.csv")


def process_email(text):
    text = text.lower()
    return list(set(text.split()))

def predict_naive_bayes(email):
    word = process_email(email)
    prediction = calculate_posterior(word)

    if (prediction['spam'] > prediction['ham']):
        return 'Spam'
    else:
        return 'Ham'

def calculate_prior():
    prior = {'spam': 0, 'ham': 0}
    spam_prob = len(emails[emails['spam'] == 1]) / len(emails)
    ham_prob = len(emails[emails['spam'] == 0]) / len(emails)
    prior['spam'] = spam_prob
    prior['ham'] = ham_prob
    return prior

def calculate_posterior(words):
    word_count_spam = defaultdict(int)
    word_count_ham= defaultdict(int)
    total_word_spam = 0
    total_word_ham = 0
    for index, row in emails.iterrows():
        if row['spam'] == 1:
            for word in row['words']:
                word_count_spam[word] += 1
                total_word_spam += 1
        else:
            for word in row["words"]:
                word_count_ham[word] += 1
                total_word_ham += 1
    prior = calculate_prior()
    spam_prob = numpy.log(prior['spam'])
    ham_prob = numpy.log(prior['ham'])

    #apply byans theorm with the leplace smoothing
    for word in words:
        word_spam_prob = (word_count_spam[word] + 1) / (total_word_spam + len(word_count_spam))
        word_ham_prob = (word_count_ham[word] + 1) / (total_word_ham + len(word_count_ham))

        spam_prob += numpy.log(word_spam_prob)
        ham_prob += numpy.log(word_ham_prob)

    return {'spam': spam_prob, 'ham': ham_prob}

def caclulate_prior(emails):
    prior = len(emails[emails['spam'] == 1]) / len(emails)
    return prior

emails["words"] = emails["text"].apply(process_email)

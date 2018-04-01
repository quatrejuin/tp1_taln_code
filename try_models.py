# Try models
# wujiechen@gmail.com
# 2018-03-01
#

import analyze
import glob
import nltk
import time
from collections import deque
from copy import copy
import pdb
from datetime import datetime
import ujson
import pickle


# Calculate the accuracy by giving the classes result list.
def cal_accuracy(class_r, test_set):
    correct = 0
    for index, c in enumerate(class_r):
        if c == test_set[index][1]:
            correct += 1
    return correct/len(class_r)


def naive_bayes_ngram_train(labeled_features,ngram=2):
    dict_classifier = {}
    for index, lemma in enumerate(labeled_features):
        if index % 10000 == 0:
            print("Index:", index)
            print("Time:", str(datetime.now()))
        train_set_w = labeled_features[lemma]
        if lemma not in dict_classifier:
            dict_classifier[lemma] = nltk.NaiveBayesClassifier.train(train_set_w)
    return dict_classifier


def naive_bayes_ngram_classify(dict_classifier, test_features, ngram=2):
    list_result = []
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(test_features[:ngram - 1])
    for index, lemma in enumerate(test_features):
        if index % 10000 == 0:
            print("Index:", index)
            print("Time:", str(datetime.now()))
        gram_wnd.append(lemma)
        if lemma in dict_classifier:
            if MULTI_VALUE_FEATURE:
                list_result += [dict_classifier[lemma].classify({"lemma" + str(i): p for i, p in enumerate(gram_wnd)})]
            else:
                list_result += [dict_classifier[lemma].classify({gram_wnd[i]: True for i, x in enumerate(gram_wnd)})]
        else:
            list_result += [lemma]
    return list_result


# Naive Bayes ngram
def try_naive_bayes_ngram(train_set_in, test_set, ngram=2):
    print("len(train_set)= {}, len(test_set)= {}".format(len(train_set_in), len(test_set)))
    train_set = copy(train_set_in)
    # Dictionary for all the features of each lemma we have seen.
    dict_features_set = {}
    # Dictionary for all the classifier of each lemma we have seen.
    dict_classifier = {}

    # extract feature
    bigrams_lemma = list(nltk.bigrams([l for l, f in train_set]))
    bigrams_lemma = [('', train_set[0][0])]+bigrams_lemma

    ngrams_lemma = list(nltk.ngrams([l for l, f in train_set],ngram))
    complete_pair = []
    for i in range(1, ngram):
        complete_pair.append(tuple([''] * (ngram - i) + [lf[0] for lf in train_set[:i]]))
    ngrams_lemma = complete_pair + ngrams_lemma

    labeled_bigrams_train_set = list(zip(ngrams_lemma, [f for l, f in train_set]))
    train_lemma_cfd = nltk.ConditionalFreqDist(train_set)
    train_inv = nltk.ConditionalFreqDist([(l, p) for p, l in labeled_bigrams_train_set])

    # find and add lemma_{j-1},lemma_j for all form_i correspond to each lemma_i
    for lemma, listf in train_lemma_cfd.items():
        if lemma not in dict_features_set:
            dict_features_set[lemma] = []
        for f in listf:
            for pair in train_inv[f]:
                if MULTI_VALUE_FEATURE:
                    dict_features_set[lemma] += [({"lemma"+str(i): p for i, p in enumerate(pair)}, f)] * train_inv[f][pair]
                else:
                    dict_features_set[lemma] += [({p: True for p in pair[:ngram]}, f)]*train_inv[f][pair]
    print("Start training... N= ", len(dict_features_set))
    print("Time:", str(datetime.now()))
    # Train for each lemma in the feature set
    dict_classifier = naive_bayes_ngram_train(dict_features_set)


    print("Test data...")
    print("Time:", str(datetime.now()))
    # Test the data, do the prediction
    list_result = naive_bayes_ngram_classify(dict_classifier, [x for x, y in test_set])

    print("Classifier accuracy percent (Naive Bayes N-gram: n={}): {}".format(ngram, cal_accuracy(list_result,
                                                                                                test_set) * 100))
    ujson.dump(list_result, open('/Users/jason.wu/Downloads/list_result_ngram.json', 'w'))
    ujson.dump(test_set, open('/Users/jason.wu/Downloads/test_set_ngram.json', 'w'))
    pickle.dump(dict_classifier, open('/Users/jason.wu/Downloads/classifier.pickle', 'wb'))
    return list_result


# Simple MLE
def try_simple_max_freq(train_set, test_set):
    cfd = nltk.ConditionalFreqDist(train_set)
    pred_class = []
    for p in test_set:
        if p[0] in cfd:
            pred_class += [cfd[p[0]].max()]
        else:
            # If it exist pas, we use the lemme as the form
            pred_class += [p[0]]
    print("Classifier accuracy percent (Simple max Freq):", cal_accuracy(pred_class, test_set) * 100)  # About 78%


# Ngram to predict wether the lemma equals to form
def try_leqf_ngram(train_set, test_set, ngram=2):
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(train_set[:ngram - 1])
    train_set_f = []
    for (lemma, forme) in train_set[ngram-1:]:
        gram_wnd.append(lemma)
        train_set_f += [({"lemma": lemma, "ngram": tuple(gram_wnd)}, lemma == forme)]
    _classifier = nltk.NaiveBayesClassifier.train(train_set_f)
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(test_set[:ngram - 1])
    test_set_f = []
    for (lemma, forme) in test_set[ngram-1:]:
        gram_wnd.append(lemma)
        test_set_f += [({"lemma": lemma, "ngram": tuple(gram_wnd)}, lemma == forme)]
    print("Classifier accuracy percent Ngram (n={}) lemma = form : {}".format(
        ngram, nltk.classify.accuracy(_classifier, test_set_f)*100))
    _classifier.show_most_informative_features(5)


# Split train and test data for the given raio
def split_train_test(_features_set, ratio=0.9):
    _train_set = _features_set[:int(len(_features_set) * ratio)]
    _test_set = _features_set[len(_train_set):]
    return _train_set, _test_set


def demo():
    global list_of_file, fl_pairs
    for file in list_of_file:
        fl_pairs += analyze.get_lemma_form_from_file(file)

    start = time.time()

    features_set = fl_pairs
    train_set, test_set = split_train_test(features_set)

    # Start predict lemma = equals
    #try_leqf_ngram(train_set, test_set, 1)
    # Ngram to predict wether the lemma equals to form n=2
    #try_leqf_ngram(train_set, test_set, 2)
    # Ngram to predict wether the lemma equals to form n=3
    #try_leqf_ngram(train_set, test_set, 3)
    # End predict lemma = equals

    print("---")
    print("Start Time: ", str(datetime.now()))

    # Simple max Freq Unigram
    #try_simple_max_freq(train_set, test_set)

    # Naive Bayes ngram n=2
    try_naive_bayes_ngram(train_set, test_set, 2)

    end = time.time()
    print("End Time:", str(datetime.now()))


def test_5_secret_files():

    test_file = "/Users/jason.wu/Desktop/IFT6285-NLP/TP1/5secrets/blind-999"

    test_set = analyze.read_test_file(test_file)

    print("---")
    print("Start Time: ", str(datetime.now()))

    # Load the classifier
    dict_classifier = pickle.load(open('/Users/jason.wu/Desktop/IFT6285-NLP/TP1/result/classifier_dev_24_241.pickle', 'rb'))
    print('Classifier loaded. Time:', time.ctime())

    print("len(test_set)= {}".format(len(test_set)))

    resultat = naive_bayes_ngram_classify(dict_classifier, [x for x, y in test_set])

    result_pair = zip(resultat, [l for l, f in test_set])

    end = time.time()
    with open(test_file+'_result', 'w', encoding='latin') as r_file:
        for f, l in result_pair:
            r_file.writelines('{}\t{}\n'.format(f, l))
    print("End Time:", str(datetime.now()))


# File name should be defined here, otherwise it tackle all the files in the folder
path = analyze.data_path_dev + 'dev-24'
list_of_file = sorted(glob.glob(path))
fl_pairs = []
MULTI_VALUE_FEATURE = True

if __name__ == '__main__':
    test_5_secret_files()

pdb.set_trace()

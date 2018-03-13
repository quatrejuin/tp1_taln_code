import analyze
import glob
import nltk
import time
from collections import deque
from copy import copy
import code
import pdb
from datetime import datetime


# Calculate the accuracy by giving the classes result list.
def cal_accuracy(class_r, test_set):
    correct = 0
    for index, c in enumerate(class_r):
        if c == test_set[index][1]:
            correct += 1
    return correct/len(class_r)


# Naive Bayes bigram
def try_naive_bayes_bigram(train_set_in, test_set):
    ngram = 2
    print("len(train_set)= {}, len(test_set)= {}".format(len(train_set_in), len(test_set)))
    train_set = copy(train_set_in)
    # Dictionary for all the features of each lemma we have seen.
    dict_features_set = {}
    # Dictionary for all the classifier of each lemma we have seen.
    dict_classifier = {}

    bigrams_lemma = list(nltk.bigrams([l for l, f in train_set]))
    bigrams_lemma = [('', train_set[0][0])]+bigrams_lemma

    bigrams_train_set = list(zip(bigrams_lemma, [f for l, f in train_set]))
    train_lemma_cfd = nltk.ConditionalFreqDist(train_set)
    bi_train_cfd_inv = nltk.ConditionalFreqDist([(l, p) for p, l in bigrams_train_set])

    # find and add lemma_{j-1},lemma_j for all form_i correspond to each lemma_i
    for lemma, listf in train_lemma_cfd.items():
        if lemma not in dict_features_set:
            dict_features_set[lemma] = []
        for f in listf:
            for pair in bi_train_cfd_inv[f]:
                dict_features_set[lemma] += [({pair[0]:True, pair[1]:True}, f)]*bi_train_cfd_inv[f][pair]
    print("Start training... N= ", len(dict_features_set))
    print("Time:", str(datetime.now()))
    # Train for each lemma in the feature set
    for index, lemma in enumerate(dict_features_set):
        if index%10000 == 0:
            print("Index:", index)
            print("Time:", str(datetime.now()))
        train_set_w = dict_features_set[lemma]
        if lemma not in dict_classifier:
            dict_classifier[lemma] = nltk.NaiveBayesClassifier.train(train_set_w)
    list_result = []
    print("Test data...")
    print("Time:", str(datetime.now()))
    # Test the data, do the prediction
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(test_set[:ngram - 1])
    for index, (lemma, form) in enumerate(test_set):
        if index%10000 == 0:
            print("Index:", index)
            print("Time:", str(datetime.now()))
        gram_wnd.append(lemma)
        if lemma in dict_classifier:
            list_result += [dict_classifier[lemma].classify({gram_wnd[0]: True, lemma: True})]
        else:
            list_result += [lemma]
    print("Classifier accuracy percent (Naive Bayes N-gram: n={}): {}".format(ngram, cal_accuracy(list_result,
                                                                                                  test_set) * 100))
    pdb.set_trace()


# Simple max Freq
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
def try_le2f_ngram(ngram, train_set, test_set):
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
def split_train_test(_features_set, ratio = 0.9):
    _train_set = _features_set[:int(len(_features_set) * ratio)]
    _test_set = _features_set[len(_train_set):]
    return _train_set, _test_set



path = analyze.data_path_dev + 'dev-24-241'
list_of_file = sorted(glob.glob(path))
fl_pairs = []

for file in list_of_file:
    fl_pairs += analyze.get_lemma_form_from_file(file)

start = time.time()


features_set = fl_pairs
train_set, test_set = split_train_test(features_set)





# Start predict lemma = equals
# Ngram to predict wether the lemma equals to form n=2
# try_le2f_ngram(2, train_set, test_set)
# Ngram to predict wether the lemma equals to form n=3
# try_le2f_ngram(3, train_set, test_set)
# End predict lemma = equals

print("---")
print("Start Time: ", str(datetime.now()))

# Simple max Freq Unigram
# try_simple_max_freq(train_set, test_set)


# Naive Bayes Bigram
try_naive_bayes_bigram(train_set, test_set)




end = time.time()
print("End Time:", str(datetime.now()))


code.interact(local=locals())
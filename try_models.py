import analyze
import glob
import pdb
import nltk
import time
from collections import deque
from copy import copy

# Calculate the accuracy by giving the classes result list.
def cal_accuracy(class_r, test_set):
    correct = 0
    for index, c in enumerate(class_r):
        if c == test_set[index][1]:
            correct += 1
    return correct/len(class_r)


# Naive Bayes N-gram
def try_naive_bayes_ngram(ngram, train_set, test_set):
    # Dictionary for all the features of each lemma we have seen.
    dict_features_set = {}
    # Dictionary for all the classifier of each lemma we have seen.
    dict_classifier = {}
    # Prepare the bigram feature set
    # Gram Window size ngram
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(train_set[:ngram - 1])
    for (lemma, form) in train_set[ngram - 1:]:
        if not lemma in dict_features_set:
            dict_features_set[lemma] = []
        gram_wnd.append(lemma)
        dict_features_set[lemma] += [({"lemma": lemma, "ngram": tuple(gram_wnd)}, form)]
    # Train for each lemma in the feature set
    for lemma in dict_features_set:
        train_set_w = dict_features_set[lemma]
        dict_classifier[lemma] = nltk.NaiveBayesClassifier.train(train_set_w)
    list_result = []
    # Test the data, do the prediction
    gram_wnd = deque([""] * ngram, maxlen=ngram)
    gram_wnd.extend(test_set[:ngram - 1])
    for lemma, form in test_set:
        gram_wnd.append(lemma)
        if lemma in dict_classifier:
            list_result += [dict_classifier[lemma].classify({"lemma": lemma, "ngram": tuple(gram_wnd)})]
        else:
            list_result += [lemma]
    print("Classifier accuracy percent (Naive Bayes N-gram: n={}): {}".format(ngram, cal_accuracy(list_result,
                                                                                                  test_set) * 100))


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



path = analyze.data_path_dev + 'dev-24'
list_of_file = sorted(glob.glob(path))
fl_pairs = []

for file in list_of_file:
    fl_pairs += analyze.analyze_single_file(file)

start = time.time()


features_set = fl_pairs
train_set, test_set = split_train_test(features_set)


# Start predict lemma = equals
# Ngram to predict wether the lemma equals to form n=2
try_le2f_ngram(2, train_set, test_set)
# Ngram to predict wether the lemma equals to form n=3
try_le2f_ngram(3, train_set, test_set)
# End predict lemma = equals

print("---")

# Simple max Freq Unigram
try_simple_max_freq(train_set, test_set)

# Naive Bayes Unigram
try_naive_bayes_ngram(1, train_set, test_set)
# Naive Bayes Bigram
try_naive_bayes_ngram(2, train_set, test_set)
# Naive Bayes N-gram n=3
try_naive_bayes_ngram(3, train_set, test_set)
# Naive Bayes N-gram n=4
try_naive_bayes_ngram(4, train_set, test_set)



end = time.time()
print("Total time in seconds:", end - start)


pdb.set_trace()
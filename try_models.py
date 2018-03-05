import analyze
import glob
import pdb
import nltk
import time
from copy import copy

# Calculate the accuracy by giving the classes result list.
def cal_accuracy(class_r, test_set):
    correct = 0
    for index, c in enumerate(class_r):
        if c == test_set[index][1]:
            correct += 1
    return correct/len(class_r)


path = analyze.data_path_dev + 'dev-24'
list_of_file = sorted(glob.glob(path))
fl_pairs = []

for file in list_of_file:
    fl_pairs += analyze.analyze_single_file(file)

#list_lemma_form = list(filter(lambda l: l[0].isalpha(), fl_pairs))
list_lemma_form = fl_pairs

list_lemma = [l for l, f in list_lemma_form]

start = time.time()

# Simple max Freq
features_set = fl_pairs
train_set = features_set[:int(len(features_set)*0.9)]
test_set = features_set[len(train_set):]
cfd = nltk.ConditionalFreqDist(train_set)
pred_class = []
for p in test_set:
    if p[0] in cfd:
        pred_class += [cfd[p[0]].max()]
    else:
        # If it exist pas, we use the lemme as the form
        pred_class += [p[0]]

print("Classifier accuracy percent (Simple max Freq):", cal_accuracy(pred_class, test_set)*100)  # About 78%


# Naive Bayes Unigram  78%
features_set = fl_pairs
train_set = features_set[:int(len(features_set)*0.9)]
test_set = features_set[len(train_set):]
train_idx = nltk.Index(train_set)


# Dictionary for all the features of each lemma we have seen.
dict_features_set = {}
# Dictionary for all the classifier of each lemma we have seen.
dict_classifier = {}

# Train for each lemma.
for lemma in train_idx:
    dict_features_set[lemma] = [({"lemma": lemma}, f) for f in train_idx[lemma]]
    train_set_w = dict_features_set[lemma]
    dict_classifier[lemma] = nltk.NaiveBayesClassifier.train(train_set_w)

list_result = []
# Test the data, do the prediction
for lemma, form in test_set:
    if lemma in dict_classifier:
        list_result += [dict_classifier[lemma].classify({"lemma": lemma})]
    else:
        list_result += [lemma]

print("Classifier accuracy percent (Naive Bayes Unigram):", cal_accuracy(list_result, test_set)*100)

pdb.set_trace()
# Naive Bayes Bigram


# POS tag
#list_lemma_tag = nltk.pos_tag(list_lemma)

# Unigram
features_set = [({"lemma": lf[0]}, lf[0] == lf[1]) for lf in list_lemma_form]


#
train_set = features_set[:int(len(features_set)*0.9)]
test_set = features_set[len(train_set):]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Classifier accuracy percent:", (nltk.classify.accuracy(classifier, test_set))*100)
print("Time:", time.time() - start)

# # With POS tag
# features_set_pos = [({"lemma": lf[0], "pos": lt[1]}, lf[0] == lf[1]) for lf, lt in zip(list_lemma_form, list_lemma_tag)]
#
# train_set = features_set_pos[:int(len(features_set_pos)*0.9)]
# test_set = features_set_pos[len(train_set):]
#
# classifier = nltk.NaiveBayesClassifier.train(train_set)
# print("Classifier accuracy percent (POS):", (nltk.classify.accuracy(classifier, test_set))*100)
# print("Time:", time.time() - start)

# Bigram
# Because the Naive Bayes assume the features are independent. So if split the lemma and lemma-1, it's not really bigram at all.
# features_set_bigram = [({"lemma": lemma, "lemma-1": fl_pairs[index][0]}, lemma == forme) for index, (lemma, forme) in enumerate(fl_pairs[1:])]
features_set_bigram = [({"lemma": lemma, "lemma-1": fl_pairs[index][0], "bigram": (fl_pairs[index][0], lemma)}, lemma == forme) for index, (lemma, forme) in enumerate(fl_pairs[1:])]
train_set = features_set_bigram[:int(len(features_set_bigram)*0.9)]
test_set = features_set_bigram[len(train_set):]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Classifier accuracy percent (Bigram):", (nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(5)

print("Time:", time.time() - start)

# Trigram
# features_set_trigram = [({"lemma": lemma, "lemma-1": fl_pairs[index][0], "lemma-2": fl_pairs[index-1][0]}, lemma==forme) for index, (lemma, forme) in enumerate(fl_pairs[2:])]
features_set_trigram = [({"lemma": lemma, "trigram": (fl_pairs[index-1][0], fl_pairs[index][0], lemma)}, lemma==forme) for index, (lemma, forme) in enumerate(fl_pairs[2:])]
train_set = features_set_trigram[:int(len(features_set_trigram)*0.9)]
test_set = features_set_trigram[len(train_set):]

classifier = nltk.NaiveBayesClassifier.train(train_set)

print("Classifier accuracy percent (Trigram):", (nltk.classify.accuracy(classifier, test_set))*100)
classifier.show_most_informative_features(5)

print("Time:", time.time() - start)


end = time.time()
print("Total time in seconds:", end - start)


pdb.set_trace()

#test_data =
#test_lemma =
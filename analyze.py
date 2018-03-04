# To analyze the TP data

import nltk, re, pprint
from nltk import word_tokenize
import re
import glob
import pdb
import json    # For print dictionary
import sys

if len(sys.argv) > 1:
    # Define the data path
    data_path_dev = sys.argv[1]


# Input a line
# Output (Forme,Lemma) pair
def splitlemma(line):
    # Split the form and lemma
    form, lemma = line.split('\t')
    # Remove the carriage return
    lemma = lemma.strip()
    return form, lemma


def analyze_single_file(fname, fixflag=False):
    total_tokens = 0
    lines = {}

    # Meta info line
    lines["meta"] = {}
    # Non-alphabetical tokens
    lines["non_alpha_form"] = {}
    # Unreadable lemmas (Non-utf8 lemma)
    list_non_utf8 = []
    # Form-Lemma pairs
    fl_pairs = []

    if fixflag:
        filefix = open(fname + '-fix', "w")

    with open(fname, encoding='latin') as file:
        for index, line in enumerate(file):
            # Read the file in latin and encode it again to use the readline but bypass the utf-8 error
            line_bytes = line.encode('latin')
            while True:
                try:
                    line = line_bytes.decode('utf-8')
                    break
                except UnicodeDecodeError as inst:
                    # After we tackle the UnicodeDecodeError it should be all fine.
                    old_char = line_bytes[inst.args[2]]
                    line_bytes = line_bytes.replace(bytes([old_char]),
                                                    bytes([old_char - (ord('a') - ord('A'))]))
                    # Finish save!
                    # Do the statistic
                    list_non_utf8 += [(index, old_char)]

            # Filter out the lines of the begin|end document meta info
            if bool(re.match("#(?:begin|end)\sdocument", line)):
                lines["meta"][index] = line
                if fixflag:
                    filefix.writelines(line)
                continue

            # Start to count the tokens
            total_tokens += 1

            form, lemma = splitlemma(line)

            # lowercase lemma
            lemma = lemma.lower()

            if not form.isalpha() and lemma.isalpha():
                lines["non_alpha_form"][index] = line

            fl_pairs += [(lemma, form)]
            if fixflag:
                filefix.writelines(form + '\t' + lemma + '\n')

    if fixflag:
        filefix.close()

    print("===")
    print(fname)
    # Show the lines statistics
    print(list((x, len(lines[x])) for x in lines))

    print("How many lines contain non-utf8:")
    print(len(nltk.Index([(x, y) for x, y in list_non_utf8])))

    print("How many kinds of first byte error non-utf8")
    print(sorted(list(nltk.Index([(hex(y), x) for x, y in list_non_utf8]))))

    print("---")
    # Start Lemma statistics
    print("Total Terms: {}".format(len(fl_pairs)))
    idx = nltk.Index(fl_pairs)
    list_lemma_freq = [(x, len(idx[x])) for x in idx]
    list_lemma_freq.sort(key=lambda x: x[1], reverse=True)
    topn = 20
    print("Lemma frequncy Top{}:".format(topn))
    print(list_lemma_freq[:topn])
    print(sum([x[1] for x in list_lemma_freq[:topn]])/len(fl_pairs))    # Top20 40.5% Top100 53%

    idx_set = nltk.Index(set(fl_pairs))
    list_form_uni = [x for x in idx_set if len(idx[x]) == 1]
    print("Lemma just has one form ratio:")  # About 50%
    print(len(list_form_uni)/len(idx_set))

    print("Lemma et form est identical:")
    print(len([x for (x, y) in fl_pairs if x == y])/len(fl_pairs))  # About 68%

    cfd = nltk.ConditionalFreqDist(fl_pairs)
    print("Show lemma 'be' show in which form and how many times for each")
    print(json.dumps(cfd['be'], ensure_ascii=False))

    list_lemma_forme_freq = [(x, len(cfd[x])) for index, x in enumerate(cfd)]
    list_lemma_forme_freq.sort(key=lambda x: x[1], reverse=True)

    return fl_pairs
    # pdb.set_trace()


# path = data_path_dev + 'dev-24'
# list_of_file = sorted(glob.glob(path))
# for file in list_of_file:
#     analyze_single_file(file)



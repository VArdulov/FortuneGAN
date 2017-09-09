import numpy as np

from csv import DictReader
from string import punctuation


fortun_csv_fn = 'data/fortune_text.csv'

punct_translator = str.maketrans('', '', punctuation)
number_translator = str.maketrans('', '', '01234567890')


def clean_text(text):
    cleaned_text = ' '. join(text.split('-')) # removes run on words
    cleaned_text = cleaned_text.translate(punct_translator).translate(number_translator).lower()
    cleaned_text = cleaned_text.strip()
    cleaned_text = ' '.join([ w for w in cleaned_text.split(' ') if w != ''])

    return cleaned_text


def collect_bigram_statistics(texts):
    counts = {'<SoS>':{}}
    for text in texts:
        text_array = text.split(' ')
        for i in range(0, len(text_array)+1):
            if (i == 0):
                counts['<SoS>'][text_array[i]] = counts['<SoS>'].get(text_array[i], 0) + 1
            else:
                if i == len(text_array):
                    if text_array[i-1] in counts:
                        counts[text_array[i-1]]['<EoS>'] = counts[text_array[i-1]].get('<EoS>', 0) + 1
                    else:
                        counts[text_array[i-1]] = {'<EoS>': 1}
                else:
                    if text_array[i-1] in counts:
                        counts[text_array[i-1]][text_array[i]] = counts[text_array[i-1]].get(text_array[i], 0) + 1
                    else:
                        counts[text_array[i-1]] = {text_array[i]: 1}
    return counts


def collect_vocabulary(texts):
    vocabulary = ['<SoS>', '<EoS>']
    for text in texts:
        for word in text.split():
            vocabulary.append(word)
    vocabulary = set(vocabulary) # leaves only unique words
    vocabulary = list(vocabulary)
    vocabulary.sort()
    return vocabulary   


def laplacian_smoothing(counts, vocabulary, k):
    distro_array = np.ones(shape=(len(vocabulary), len(vocabulary))) * k
    for w1 in counts:
        for w2 in counts[w1]:
            i_w1  = vocabulary.index(w1)
            i_w2  = vocabulary.index(w2)
            distro_array[i_w1, i_w2] += counts[w1][w2]

    for i in range(0, distro_array.shape[0]):
        distro_array[i, :] /= np.sum(distro_array[i, :])

    return distro_array

def good_turing_smoothing(counts, vocabulary):
    # TODO: implement good_turing_smoothing !figure out a good way to line fit and then normalize!!
    pass

if __name__ == '__main__':

    fortunes = []

    with open(fortun_csv_fn, 'r') as csv_file:
        reader = DictReader(csv_file)
        for row in reader:
            if 'fortune_text' in row:
                fortunes.append(row['fortune_text'])

    print("Texts loaded:", len(fortunes), 'total fortunes')
    print('Clearning text')
    for i in range(0, len(fortunes)):
        fortunes[i] = clean_text(fortunes[i])

    print('Text \"clearned\" (punctionation removed, capital letters removed, stripped for extra spaces)')
    print('Collecting unsmoothed bi-gram statistics')
    bigram_counts = collect_bigram_statistics(fortunes)
    keys = list(bigram_counts.keys())
    for key in keys[0:5]:
        print(key, bigram_counts[key])

    vocabulary = collect_vocabulary(fortunes)
    print('There are', len(vocabulary), 'words in our vocabulary')
    print(vocabulary[0:10])

    lp_smoothed_bigram_probabilities = laplacian_smoothing(bigram_counts, vocabulary, 1)
    for i in range(0, 100):
        for j in range(0, 100):
            print('P('+vocabulary[j]+'|'+vocabulary[i]+')=\t', lp_smoothed_bigram_probabilities[i, j])


    # gt_smoothed_bigram_counts = good_turing_smoothing(bigram_counts, vocabulary)








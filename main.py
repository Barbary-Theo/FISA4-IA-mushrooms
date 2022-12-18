import os
import pathlib
from collections import Counter

import keras as keras
from keras.models import Sequentialx
from keras.layers import Dense
from keras import preprocessing
from numpy import array, zeros

# import matplotlib.pyplot as plt

def get_veneneux_comestible_from_csv_file(file_name):
    veneneux = []
    commestible = []
    header = []

    with open(file_name + ".csv", "r") as file:

        lines = file.read().split("\n")
        index = 0

        for line in lines:
            if index != 0:
                if line[0].__contains__("e"):
                    commestible.append(line)
                else:
                    veneneux.append(line)
            else:
                header.append(line)
            index += 1

    with open(file_name + "-veneneux.csv", "w") as f:
        f.write(header.__str__() + "\n")
        for line in veneneux:
            print(line)
            f.write(line + "\n")

    with open(file_name + "-comestibles.csv", "w") as f:
        f.write(header.__str__() + "\n")
        for line in commestible:
            print(line)
            f.write(line + "\n")

    return veneneux, commestible

def load_doc(filename: str) -> list:
    """retourne les lignes de texte du fichier filename"""
    # open the file as read only
    with open(filename) as file:
        text = file.read().splitlines()
    return text

def add_doc_to_vocab(filename, vocab):
    """cumule dans la liste vocab les mots du fichier filename
    (1 seule occurence par mot dans vocab)"""
    # load doc
    lines = load_doc(filename)
    doc = ''.join(lines)
    doc.replace(',', ' ',-1)
    #print("doc:" + doc)
    # update counts
    vocab.update(doc)
    del vocab[',']
    del vocab['?']
    del vocab["'"]
    del vocab['-']
    del vocab['[']
    del vocab[']']

def create_global_vocabulary():
    """creer un vocabulaire (liste de mots clés associés à leurs occurences)
    par la suite, un mot d'un texte ne faisant pas partie du vocabulaire ne sera
    pas compte"""
    vocab = Counter()
    # ajouter les mots cles des fichiers
    add_doc_to_vocab(str(pathlib.Path().resolve()) + '/mushrooms-comestibles.csv', vocab)
    add_doc_to_vocab(str(pathlib.Path().resolve()) + '/mushrooms-veneneux.csv', vocab)

    # afficher le nb de mots cles trouves
    print("nb de mots cles trouves dans les repertoires : ", len(vocab))
    print("les 10 premiers mots cles du vocabulaire \
    (et leur nb d\'apparition dans les exemples)  : \n", end='')
    i = 0
    for (mot, count) in vocab.items():
        if mot != ',' and mot != "'":
            print('mot:', mot, 'count:', count, end=", ")
        i = i + 1
        if i > 10: break
    # afficher les 10 mots cles les plus utilises
    print("\nles 10 mots cles les plus utilises : ", vocab.most_common(10))

    # ne garder que les mots clés apparaissant au moins 2 fois
    min_occurrence = 2
    tokens = [token for (token, count) in vocab.items() if count >= min_occurrence]
    print('en otant les mots utilise moins de ', min_occurrence, ' fois,',
          ' nb de mots cles = ', len(tokens))
    return tokens

def filter_line(line, vocab)->list:
    """retourne la liste des mots cles de la ligne appartenant au vocabulaire vocab"""
    # clean line
    tokens = line
    # filter by vocab
    tokens = [token for token in tokens if token in vocab]

def process_train_file(filenametrain, vocab)->list:
    """retourne deux listes des mots cles du repertoire directory;
    la 1ere liste est issu de 90% du fichier d'entrainement,
    la 2e du fichier des 10% restant"""
    lines_train = list()
    lines_test = list()
    i=1
    lines = list()
    # load and clean the file
    with open(filenametrain) as file:
        lines = file.read().splitlines()
    filtered_lines = [filter_line(line, vocab) for line in lines]
    for i in range(0,int(len(filtered_lines) * 9/10)):
        lines_train.append(filtered_lines[i])
    for j in range(int(len(filtered_lines) * 9/10),len(filtered_lines)):
        lines_test.append(filtered_lines[j])
    return (lines_train,lines_test)

def main():
    result = get_veneneux_comestible_from_csv_file("mushrooms")

    print(f"Vénéneux : {len(result[0])}, comestibles : {len(result[1])}")
    print(create_global_vocabulary())

    vocab_filename = 'mushrooms.csv'
    vocab = load_doc(vocab_filename)
    vocab = set(vocab)
    # load training and testing examples
    (positive_lines_train, positive_lines_test) = process_train_file('mushrooms-comestibles.csv', vocab)
    (negative_lines_train, negative_lines_test) = process_train_file('mushrooms-veneneux.csv', vocab)
    # summarize what we have
    print("nb exemples d'entrainement positifs : ", len(positive_lines_train))
    print("nb exemples d'entrainement negatifs : ", len(negative_lines_train))
    print("nb exemples de tests positifs : ", len(positive_lines_test))
    print("nb exemples de tests negatifs : ", len(negative_lines_test))

    # create the tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    # build the training doc based on training data
    training_doc = list()
    training_doc.extend(positive_lines_train)
    training_doc.extend(negative_lines_train)
    # ask the tokenizer to build the bag of words : a set of (word, use)
    tokenizer.fit_on_texts(training_doc)
    xTrain = tokenizer.texts_to_matrix(training_doc, mode='freq')
    ##TODO: regarder les autres modes que freq; par exemple 'bin', qu'apportent-ils ?
    print('Xtrain contient ', xTrain.shape[0], ' exemples de ', xTrain.shape[1], ' valeurs')
    print('une valeur = fréquence d\'apparition des mots dans le vocabulaire global.')
    print('Ainsi, premier exemple d\'entrainement = \n', xTrain[0])

if __name__ == "__main__":
    main()
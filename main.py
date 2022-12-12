import os
import pathlib
from collections import Counter

#import keras as keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras import preprocessing
#from numpy import array, zeros

#import matplotlib.pyplot as plt


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



def load_doc(filename:str)->list:
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
    doc = ' '.join(lines)
    print(doc)
    # update counts
    if doc != ',':
        vocab.update(doc)

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
    i=0
    for (mot,count) in vocab.items():
        print(mot,':',count,end=", ")
        i = i+1
        if i>10:break
    # afficher les 10 mots cles les plus utilises
    print("\nles 10 mots cles les plus utilises : ", vocab.most_common(10))

    # ne garder que les mots clés apparaissant au moins 2 fois
    min_occurrence = 2
    tokens = [token for (token,count) in vocab.items() if count >= min_occurrence]
    print('en otant les mots utilise moins de ', min_occurrence, ' fois,',
          ' nb de mots cles = ',len(tokens))
    return tokens



def main():

    result = get_veneneux_comestible_from_csv_file("mushrooms")

    print(f"Vénéneux : {len(result[0])}, comestibles : {len(result[1])}")
    print(create_global_vocabulary())


if __name__ == "__main__":
    main()

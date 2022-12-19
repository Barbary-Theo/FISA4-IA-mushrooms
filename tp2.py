##import pour les fichiers et le traitement de données :
import csv
import os
from datetime import datetime

import nltk
from collections import Counter
##import pour les réseaux de neurones :
import keras as keras
import tensorflow
from keras import Sequential
from keras.layers import Dense
from keras import preprocessing
from numpy import array, zeros
from colorama import Fore, Back, Style

import matplotlib.pyplot as plt

if (str(os.path).__contains__("basil")):
    rep = 'C:/Users/basil/PycharmProjects/tp1ia/'  ## ICI VOTRE REPERTOIRE DE TRAVAIL
else:
    rep = "/Users/theobabrary/Desktop/Cours/S7/IA/ADAM/FISA4-IA-mushrooms/"
# le code suivant charge l'ensemble des mots non importants
nltk.download("stopwords")


def load_doc(filename: str) -> list:
    """retourne les lignes de texte du fichier filename"""
    # open the file as read only
    with open(filename, encoding="utf-8") as file:
        text = file.read().splitlines()
    return text


def clean_doc(doc) -> list:
    """retourne la liste de mots clés inclus dans le texte doc
    qui ne font pas parti des stop_words anglais et francais
    retire d autres mots cles comme 'vers', 'lors' , .."""
    tokens = doc.lower()
    tokens = tokens.split()
    # remove all stop words
    stop_words = set(nltk.corpus.stopwords.words('english'))
    french_stop_words = set(nltk.corpus.stopwords.words('french'))
    stop_words = stop_words.union(french_stop_words)
    tokens = [token for token in tokens if token not in stop_words]

    # remove all punctuation
    tokens = [token for token in tokens if token.isalpha()]
    # remove all short tokens
    tokens = [token for token in tokens if len(token) > 0 or token == "IA" or token == "AI"]

    return tokens


def test_recup():
    # mots clés du fichier des news reelles
    filename = rep + 'reellesnews.txt'
    lines = load_doc(filename)
    text = ' '.join(lines)
    tokens = clean_doc(text)
    print('les 10 premiers mots cles de ', filename)
    print(tokens[:10])
    # mots clés du fichier des fake news
    filename = rep + 'fakenews.txt'
    lines = load_doc(filename)
    text = ' '.join(lines)
    tokens = clean_doc(text)
    print('les 10 premiers mots cles de ', filename)
    print(tokens[:10])


def add_doc_to_vocab(filename, vocab):
    """cumule dans la liste vocab les mots du fichier filename
    (1 seule occurence par mot dans vocab)"""
    # load doc
    lines = load_doc(filename)
    doc = ' '.join(lines)
    # clean doc
    tokens = clean_doc(doc)

    # update counts
    vocab.update(tokens)


def create_global_vocabulary():
    """creer un vocabulaire (liste de mots clés associés à leurs occurences)
    par la suite, un mot d'un texte ne faisant pas partie du vocabulaire ne sera
    pas compte"""
    vocab = Counter()
    # ajouter les mots cles des fichiers
    add_doc_to_vocab(rep + 'comestible.txt', vocab)
    add_doc_to_vocab(rep + 'poisonous.txt', vocab)

    # afficher le nb de mots cles trouves
    print("nb de mots cles trouves dans les repertoires : ", len(vocab))
    print("les 10 premiers mots cles du vocabulaire \
    (et leur nb d\'apparition dans les exemples)  : \n", end='')
    i = 0
    for (mot, count) in vocab.items():
        print(mot, ':', count, end=", ")
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


## Sauvegarde du vocabulaire
def save_list(lines, filename):
    """sauve les mots de la liste lines dans le fichier filename"""
    # open file
    file = open(filename, 'w', encoding="utf-8")
    data = '\n'.join(lines)
    # write text
    file.writelines(data)
    # close file
    file.close()


def csv_to_txt(csv_file, reel_text_file, fake_text_file):
    with open(csv_file, 'r', encoding="utf8") as csvfile, open(reel_text_file, 'w',
                                                               encoding="utf8") as reeltextfile, open(fake_text_file,
                                                                                                      'w',
                                                                                                      encoding="utf8") as faketextfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        i = 1
        for row in spamreader:
            if i > 5000:
                return
            i += 1

            #split row

            if row[0] == "p":
                faketextfile.write(str(row[1:]).replace(",","").replace("[","").replace("]","").replace("'","") + "\n")
            else:
                reeltextfile.write(str(row[1:]).replace(",","").replace("[","").replace("]","").replace("'","") + "\n")


def filter_line(line, vocab) -> list:
    """retourne la liste des mots cles de la ligne appartenant au vocabulaire vocab"""
    # clean line
    tokens = clean_doc(line)
    # filter by vocab
    tokens = [token for token in tokens if token in vocab]
    return ' '.join(tokens)


def process_train_file(filenametrain, vocab) -> list:
    """retourne deux listes des mots cles du repertoire directory;
    la 1ere liste est issu de 90% du fichier d'entrainement,
    la 2e du fichier des 10% restant"""
    lines_train = list()
    lines_test = list()
    i = 1
    lines = list()
    # load and clean the file
    with open(filenametrain, encoding="utf-8") as file:
        lines = file.readlines()
    print("nb de lignes dans le fichier d'entrainement : ", len(lines))
    filtered_lines = [filter_line(line, vocab) for line in lines]
    # split into train and test
    split = int(len(filtered_lines) * 0.9)

    lines_train = filtered_lines[:split]
    lines_test = filtered_lines[split:]

    return (lines_train, lines_test)


def process_all_file():
    loss = []
    val_loss = []
    acc = []
    val_acc = []

    csv_to_txt('mushrooms.csv', 'poisonous.txt', 'comestible.txt')

    ## A la 1ere utilisation, et a chaque modification des fichiers de données
    tokens = create_global_vocabulary()
    save_list(tokens, 'vocabMushroom.txt')

    # SI le vocabulaire  n'a ete cree dans cette session mais avant ALORS le charger
    # load the vocabulary
    vocab_filename = 'vocabMushroom.txt'
    vocab = load_doc(vocab_filename)
    vocab = set(vocab)
    # load training and testing examples
    (positive_lines_train, positive_lines_test) = process_train_file(rep + 'comestible.txt', vocab)
    (negative_lines_train, negative_lines_test) = process_train_file(rep + 'poisonous.txt', vocab)

    # create the tokenizer
    tokenizer = keras.preprocessing.text.Tokenizer()
    # build the training doc based on training data
    training_doc = list()
    training_doc.extend(positive_lines_train)
    training_doc.extend(negative_lines_train)
    # ask the tokenizer to build the bag of words : a set of (word, use)*
    tokenizer.fit_on_texts(training_doc)

    xTrain = tokenizer.texts_to_matrix(training_doc, mode='freq')

    # ytrain = suite de (0 (classement pour eval positive), 1 (classements pour éval négative))
    yTrain = zeros(len(positive_lines_train) + len(negative_lines_train))
    yTrain[:len(positive_lines_train)] = 1

    # build the test doc by alternating positive lines and negative lines
    test_Doc = list()
    test_Doc.extend(positive_lines_test)
    test_Doc.extend(negative_lines_test)
    # ask to the tokenizer to give the bag of words : a set of (word, frequence of use),
    # the words are already kown by the tokenizer*
    xTest = tokenizer.texts_to_matrix(test_Doc, mode='binary')
    # print('Xtest contient ', xTest.shape[0], ' exemples de ', xTest.shape[1], ' valeurs de fréquence.')

    # sortie attendues des exemples de test, ytest = suite de (0, 1)
    yTest = zeros(len(positive_lines_test) + len(negative_lines_test))
    yTest[:len(positive_lines_test)] = 1

    # TODO: définir la structure du réseau
    model = keras.Sequential()
    """
    # 1
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='elu'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    log_dir = "log/1/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), batch_size=500, epochs=20, verbose=2,callbacks=[tensorboard_callback])
    """
    """
    #2
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(1, activation='relu'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    log_dir = "log/2/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=20, batch_size=500, verbose=2,callbacks=[tensorboard_callback])
    """

    """
    #3
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='elu'))

    # model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    # model.add(Dense(1, activation='relu'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/3/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=100,batch_size=500, verbose=2,callbacks=[tensorboard_callback])
"""
    """
    #4
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(10, activation='elu'))

    # model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    # model.add(Dense(1, activation='relu'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/4/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=20, verbose=2,batch_size=500,callbacks=[tensorboard_callback])
    """
    """
    #5

    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(10, activation='relu'))

    # model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    # model.add(Dense(1, activation='relu'))
    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/5/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=20, batch_size=500,verbose=2,callbacks=[tensorboard_callback])
    """
    """
    #6
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(10, activation='elu'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/6/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=20, verbose=2,batch_size=500,callbacks=[tensorboard_callback])

    """
    """

    #7
    model.add(Dense(1000, activation='relu'))

    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/7/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=10, verbose=2,batch_size=500,callbacks=[tensorboard_callback])
    """
    """
    #8
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(10, activation='relu'))

    # model.add(Dense(50, input_shape=(n_words,), activation='relu'))
    # model.add(Dense(1, activation='relu'))
    model.compile(loss='MSE', optimizer='sgd', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/8/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=20,batch_size=500, verbose=2,
                        callbacks=[tensorboard_callback])
    """

    #9
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='elu'))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])

    # TODO tester differents nb de tests (epochs)
    log_dir = "log/m9/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=50,batch_size=500, verbose=2,
                        callbacks=[tensorboard_callback])

    """

    # 10

    model.add(Dense(400, activation='elu'))
    model.add(Dense(400, activation='elu'))
    model.add(Dense(400, activation='elu'))
    model.add(Dense(400, activation='relu'))

    model.compile(loss='MSE', optimizer='adam', metrics=['accuracy'])
    
    # TODO tester differents nb de tests (epochs)
    log_dir = "log/10/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tensorflow.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=50, batch_size=500, verbose=2,
                        callbacks=[tensorboard_callback])
    """
    history_dict = history.history
    history_dict.keys()
    # add the loss and val_loss to the list
    loss.extend(history_dict['loss'])
    val_loss.extend(history_dict['val_loss'])
    acc.extend(history_dict['accuracy'])
    val_acc.extend(history_dict['val_accuracy'])

    i = 0

    epochs = range(1, len(loss) + 1)

    plt.plot(epochs, loss, 'b-*', label='erreur sur exemples d\'apprentissage')
    # draw the accuracy evolution in blue
    plt.plot(epochs, val_loss, 'r-*', label='erreur sur exemples de validation')
    plt.title('Erreur')
    plt.xlabel('Epochs')
    plt.ylabel('Erreur')
    plt.legend()

    plt.show()

    # draw the loss evolution in blue
    plt.plot(epochs, acc, 'b-*', label='précision sur exemples d\'apprentissage')
    # draw the accuracy evolution in blue
    plt.plot(epochs, val_acc, 'r-*', label='précision sur exemples de validation')
    plt.title('précision')
    plt.xlabel('Epochs')
    plt.ylabel('précision')
    plt.legend()

    plt.show()

    # evaluate
    loss, acc = model.evaluate(xTest, yTest, verbose=0)
    print('Précision sur exemples de test : %f' % (acc * 100))


if __name__ == '__main__':
    process_all_file()

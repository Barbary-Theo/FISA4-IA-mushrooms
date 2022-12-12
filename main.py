import tensorboard
import keras as keras
from keras.models import Sequential
from keras.layers import Dense
from keras import preprocessing
from numpy import array, zeros

import matplotlib.pyplot as plt



def read_csv_file(file_name):

    veneneux = []
    commestible = []


    with open(file_name, "r") as file:
        for line in file.read():
            print(line)


def main():
    read_csv_file("mushrooms.csv")


if __name__ == "__main__":
    main()

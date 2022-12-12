#import keras as keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras import preprocessing
#from numpy import array, zeros

#import matplotlib.pyplot as plt



def read_csv_file(file_name):

    veneneux = []
    commestible = []

    with open(file_name, "r") as file:

        lines = file.read().split("\n")
        index = 0

        for line in lines:
            if index != 0:
                if line[0].__contains__("e"):
                    commestible.append(line)
                else:
                    veneneux.append(line)
            index += 1


    return veneneux, commestible


def main():

    result= read_csv_file("mushrooms.csv")

    print(f"Veneer : {len(result[0])}, comestibles : {len(result[1])}")


if __name__ == "__main__":
    main()

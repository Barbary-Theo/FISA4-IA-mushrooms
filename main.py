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
            f.write(line + "\n")

    with open(file_name + "-comestibles.csv", "w") as f:
        f.write(header.__str__() + "\n")
        for line in commestible:
            f.write(line + "\n")

    return veneneux, commestible


def main():

    result = get_veneneux_comestible_from_csv_file("mushrooms")

    print(f"Vénéneux : {len(result[0])}, comestibles : {len(result[1])}")


if __name__ == "__main__":
    main()

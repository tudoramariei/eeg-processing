#!/usr/bin/env python

letters_list = [
    'Fp', 'AF', 'F',
    'Fc', 'C', 'FT',
    'T', 'CP', 'TP',
    'P', 'PO', 'O']


def get_all_electrodes():
    list_of_letters = []
    for letter in letters_list:
        for number in range(0, 11):
            if number == 0:
                list_of_letters.append("{0}{1}".format(letter, 'z'))
            else:
                list_of_letters.append("{0}{1}".format(letter, number))
    return list_of_letters


def get_empotiv_epoc_electrodes():
    list_of_letters = []

    for letter in letters_list:
        for number in range(0, 11):
            if (((letter == 'P' or letter == 'T' or letter == 'F') and
                    (number == 7 or number == 8)) or

                ((letter == 'AF' or letter == 'F') and
                    (number == 3 or number == 4)) or

                ((letter == 'FC') and
                    (number == 5 or number == 6)) or

                ((letter == 'O') and
                    (number == 1 or number == 2))):
                list_of_letters.append("{0}{1}".format(letter, number))
    return list_of_letters


list_of_letters = []
list_of_letters = get_all_electrodes()

for element in list_of_letters:
    print(element)

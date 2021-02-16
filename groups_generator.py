import random


def generate_groups():

    groups = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    random.shuffle(digits)
    for i in range(2):
        for j in range(5):
            groups[i][j] = digits[j+(i*5)]
    return groups

import numpy as np


def read__train__test():
    n = 943
    m = 1682

    train_set = np.zeros((n, m), dtype=np.float)

    yui = np.zeros((n, m), dtype=np.float)

    train_exp_set = []

    #  用户评分物品集合 区分各种评分 1， 2， 3， 4
    train_im_set = {}

    test_set = []

    file_train = ''
    file_test = ''

    try:
        file_train = open('100k_train.base', 'r')
        for line in file_train.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            train_set[uid - 1][iid - 1] = r
            yui[uid - 1][iid - 1] = 1
            record = [uid-1, iid-1, r]
            train_exp_set.append(record)
            rating_dic = {1: [], 2: [], 3: [], 4: [], 5: []}
            train_im_set.setdefault(uid-1, rating_dic)
            train_im_set[uid-1][r].append(iid-1)

        file_test = open('100k_test.test', 'r')
        for line in file_test.readlines():
            line = line.strip()
            ss = line.split()
            uid = int(ss[0])
            iid = int(ss[1])
            r = int(ss[2])
            record = [uid-1, iid-1, r]
            test_set.append(record)

    finally:
        if file_train:
            file_train.close()
        if file_test:
            file_test.close()

    return train_set, yui, train_exp_set, train_im_set, test_set

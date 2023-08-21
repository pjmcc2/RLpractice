import numpy as np
from numpy.random import default_rng



def gen_track(w, h, seed=None):
    out = []
    # make approx. 1/6th of the track the starting turn
    for i in range(h // 6 + 1):
        # the back 5th should be blocked off to make it a "turn"
        out.append(gen_start_row(w, w // 5))
    # rest of the rows
    for i in range(h - h // 6 - 1):
        out.append(gen_row(w, prev_row=out[-1]))

    return np.array(out)


def gen_row(length, size=None, prev_row=None):
    if size is None:
        num1s = rng.integers(length // 2 - 1, length // 2 + 1)
    else:
        num1s = size
    row = np.zeros(length)

    if prev_row is not None:
        connection = False
        for i in range(num1s):
            row[i] = 1
            if i % 2 == 0:
                row[int(-i / 2)] = 1
            connection = connection or (prev_row[i] + row[i] == 0)
        if not connection:
            id = len(row) // 2
            iter = 0
            row[id] = 0
            while not connection:
                try:
                    if prev_row[id + iter] + row[id + iter] == 0 or prev_row[id - iter] + row[id - iter] == 0:
                        connection = True
                    else:
                        row[id + iter] = 0
                        row[id - iter] = 0
                    iter += 1
                except:
                    break  # TODO FIX
            # for i in range(len)
            # make smooth goes here
    else:
        for i in range(num1s):
            row[i] = 1
    return row


def gen_start_row(w, num):
    row = np.zeros(w)
    for i in range(1, num + 1):
        row[-i] = 1
    return row
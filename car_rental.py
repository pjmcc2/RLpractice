import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson

from numpy.random import default_rng


def init_actions(size1, size2, move_limit):
    rng = default_rng()
    out = []
    for i in range(size1):
        t = []
        for j in range(size2):
            max1 = np.min((i, size2 - j,
                           move_limit))  # check if there are more cars to move than limit/space available in other lot
            max2 = np.min((size1 - i, j, move_limit))
            if max1 == 0:
                l1 = 0
            else:
                l1 = rng.integers(0, max1 + 1)
            if max2 == 0:
                l2 = 0
            else:
                l2 = rng.integers(0, max2 + 1)
            t.append((l1, l2))
        out.append(t)
    return out


def get_actions(m, n, max):
    out = []
    for i in range(max + 1):  # 0 - max
        if i > m:
            break
        for j in range(max + 1):
            if j > n:
                break
            out.append((i, j))
    return out


def eval_state(state, values, policy, distributions, discount=0.9):
    assert len(values.shape) > 1
    sum = 0
    o1, o2 = state  # original/old state
    try:
        t1, t2 = policy[o1][o2]  # transfers
    except:
        t1, t2 = policy
    cost = 2 * (t1 + t2)
    c1 = o1 - t1 + t2  # cars in lot 1
    c2 = o2 - t2 + t1  # cars in lot 2

    max_val = np.max(values.shape)

    dist1, dist2 = distributions

    for l1i in range(values.shape[0]):
        for l2i in range(values.shape[1]):
            # next_state = (l1i, l2i)
            diff1 = l1i - c1
            diff2 = l2i - c2

            # Create list of probability,reward tuples
            plist1 = []
            plist2 = []
            for i in range(max_val):  # try for all possible values (if not possible its 0)
                if i + diff1 < 0:  # avoid negative indexing
                    p1 = 0
                    r1 = 0
                    plist1.append((p1, r1))
                elif i <= c1:
                    p1 = dist1[i, i + diff1]
                    r1 = 10 * i
                    plist1.append((p1, r1))

                if i + diff2 < 0:
                    p2 = 0
                    r2 = 0
                    plist2.append((p2, r2))
                elif i <= c2:
                    p2 = dist2[i, i + diff2]
                    r2 = 10 * i
                    plist2.append((p2, r2))

            # add to total sum for each pair in the lists
            # cancer inducing btw readers beware
            for j in range(len(plist1)):
                for k in range(len(plist2)):
                    sum += plist1[j][0] * plist2[k][0] * (plist1[j][1] + plist2[k][1] + discount * values[l1i, l2i])

    return sum - cost


############### LOOP THROUGH ALL STATES, EVALUATE, AND UPDATE ################
def policy_eval(values, policy, distributions, eval_func, acc_min=0.001, max_iters=50):
    min = acc_min  # numerical accuracy
    num_iters = 0
    delta = 1
    while delta >= min:
        delta = 0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                oldv = values[i, j]
                values[i, j] = eval_func((i, j), values, policy, distributions)
                delta = np.amax((delta, np.abs(oldv - values[i, j])))
        num_iters += 1
        if num_iters >= max_iters:
            print(f"Breaking after {max_iters} iterations.")
            break
    print(f"Finished after {num_iters} iterations.")


def improve_policy(policy, values, distributions, eval_func):
    stable = True
    for i in range(len(policy)):  # number of lists (rows)
        print(f"Progress:{i}/{len(policy)}")
        for j in range(len(policy[0])):  # length of a sub list: number of cols
            old_p = policy[i][j]

            valid_actions = get_actions(i, j, 5)
            exp_r_list = [eval_func((i, j), values, a, distributions) for a in valid_actions]
            new_p = valid_actions[np.argmax(exp_r_list)]
            stable = new_p == old_p and stable
            policy[i][j] = new_p
    return stable


######## Policy iteration for jack's car rental! ###################
def policy_iteration(lotsize1, lotsize2, max_movement, max_iters=50, altered=False):
    #### initialize ####
    rng = default_rng()
    # v = rng.standard_normal((lotsize1,lotsize2))
    v = rng.standard_exponential((lotsize1, lotsize2))
    policy = init_actions(lotsize1, lotsize2, max_movement)

    dist1 = np.outer(poisson.pmf(np.arange(lotsize1 + 1), 3), poisson.pmf(np.arange(lotsize1 + 1), 3))
    dist2 = np.outer(poisson.pmf(np.arange(lotsize1 + 1), 4), poisson.pmf(np.arange(lotsize1 + 1), 2))

    dists = (dist1,dist2)
    stable = False
    iters = 0
    while not stable:
        print(f"Evaluating Policy Loop:{iters + 1}")
        policy_eval(v, policy, dists, eval_state)  # messed up my structure here, how do i give it a different max value?
        print(f"Improving Policy Loop:{iters + 1}")
        stable = improve_policy(policy, v, dists, eval_state)
        iters += 1
        if iters >= max_iters:
            break
    return v, policy


if __name__ == "__main__":
    vs, pis = policy_iteration(21, 21, 5)
    arrayd = np.array(pis)
    arrayd[:, :, 1] *= -1
    fary = arrayd[:, :, 0] + arrayd[:, :, 1]
    contours = plt.contour(np.arange(10), np.arange(10), fary)
    plt.clabel(contours, inline=1, fontsize=10)
    plt.show()
    print(fary)
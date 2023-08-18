import numpy as np
from numpy.random import default_rng

rng = default_rng(seed=1000)  # 3432


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


def gen_run(policy, t, training=True, max_iters=500):
    terminal = False
    num_iters = 0
    i = rng.choice(np.arange(t.shape[1] // 6 + 1))
    j = 0
    v1 = 0
    v2 = 0
    try:
        a = policy[(i, j, v1, v2)]
    except:
        a = policy(0)
    history = [((i, j, v1, v2), a)]
    while not terminal:
        try:
            a = policy[(i, j, v1, v2)]
        except:
            a = policy(0)
        if training:
            if rng.uniform() <= 0.1:
                a = (0, 0)
        history.append(((i, j, v1, v2), a, -1))
        i, j, v1, v2, terminal = update(i, j, v1, v2, a, t)
        num_iters += 1
        if num_iters == max_iters:
            terminal = True
    return history


def update(i, j, v1, v2, action, track):
    v1 += action[0] if v1 + action[0] >= 0 else 0
    v2 += action[1] if v2 + action[1] >= 0 else 0
    if v2 == v1 and v1 == 0 and j != 0:
        i = rng.choice(np.arange(track.shape[1] // 6 + 1))
        j = 0
        v1, v2 = 0, 0
        return i, j, v1, v2, False
    if v1 >= 5:
        v1 = 4
    if v2 >= 5:
        v2 = 4
    i += v1
    j += v2
    safe, terminal = is_trajectory_safe(i, j, v1, v2, track)
    if not safe:
        i = rng.choice(np.arange(track.shape[1] // 6 + 1))
        j = 0
        v1, v2 = 0, 0
    return i, j, v1, v2, terminal


def is_trajectory_safe(v, h, t1, t2, track):
    # rule for checking: move down then right as much as possible
    # Kinda bad rule but will work for this implementation (e.g. diagonals arent safe)
    t1_step = 0
    t2_step = 0
    while (t1_step != t1) or (t2_step != t2):
        if t1_step + 1 <= t1:
            t1_step += 1
            try:
                if track[v + t1_step, h + t2_step] == 1:
                    return False, False
                elif track[v + t1_step, h + t2_step] == 0 and v + t1_step == track.shape[0] - 1:
                    return True, True
            except IndexError:
                if v + t1_step >= track.shape[0]:
                    return True, True
                return False, False
        if t2_step + 1 <= t2:
            t2_step += 1
            try:
                if track[v + t1_step, h + t2_step] == 1:
                    return False, False
            except IndexError:
                return False, False
    return True, False


def init_pol(states):
    return {state: tuple(rng.choice([-1, 0, 1], size=2)) for state in states}


def unif_pol(dummy):
    return tuple(rng.choice([-1, 0, 1], size=2))


if __name__ == "__main__":
    # setup stuff

    track = gen_track(5, 5)
    states = []
    for i in range(track.shape[0]):
        for j in range(track.shape[1]):
            for k in range(5):
                for l in range(5):
                    states.append((i, j, k, l))

    start_policy = init_pol(states)
    action_list = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1)]
    q_list = []
    for s in states:
        for a in action_list:
            q_list.append((s, a))
    q = {pair: rng.uniform() for pair in q_list}
    c = {pair: 0 for pair in q_list}
    pi = {state: action_list[np.argmax([q[(state, a)] for a in action_list])] for state in states}

    num_episodes = 10000
    episodes = 0
    while episodes != num_episodes:
        episodes += 1
        run = gen_run(unif_pol, track)
        g = 0
        w = 1
        for i in range(1, len(run) - 1):
            state, action, reward = run[-i]
            g += reward if reward is not None else 0
            c[(state, action)] += w
            q[(state, action)] += w / c[(state, action)] * (g - q[(state, action)])
            pi[state] = action_list[np.argmax([q[(state, a)] for a in action_list])]  # need to do the close thing?
            if action != pi[state]:
                break
            w = 9 * w  # (1/(1/9))  before or after or during if?

    run2 = gen_run(pi, track, training=False)
    print([(step[0][0], step[0][1]) for step in run2])

import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt

rng = default_rng()


def stable_track(height=12, width=12, mode="blank"):
    if mode == "blank":
        return np.zeros((width, height))
    elif mode == "right":
        out = []
        for i in range(height // 3):
            out.append([0 for j in range(width)])
        for k in range(height // 3, height):
            out.append([1 if i < width // 3 else 0 for i in range(width)])
        return np.array(out)
    elif mode == "left":
        out = []
        for i in range(height // 3):
            out.append([0 for j in range(width)])
        for k in range(height // 3, height):
            out.append([1 if i >= 2 * width // 3 else 0 for i in range(width)])
        return np.array(out)


def gen_run(policy, track, training=True, verbose=False, max_iters=500):
    terminal = False
    num_iters = 0
    # random starting row
    i = rng.choice(np.arange(track.shape[1] // 6 + 1))
    # starting column is 0 for left starts or -1 for right
    j = 0
    # starting velocities are 0
    v1 = 0
    v2 = 0
    reward = -1
    noise_chance = 0.1
    restarted = True
    try:
        a = policy[(i, j, v1, v2)]
    except KeyError:
        a = policy(0)
    # Starting state, action
    # history = [((i, j, v1, v2), a, 0)]  # remove 0?
    history = []
    while not terminal:
        try:
            a = policy[(i, j, v1, v2)]
        except KeyError:
            a = policy(0)  # optional change?
        if training:
            if rng.uniform() <= noise_chance:
                a = (0, 0)
        history.append(((i, j, v1, v2), a, reward))
        if verbose:
            print(history[-1])
        i, j, v1, v2, terminal, restarted = update(i, j, v1, v2, a, track, restarted, verbose=verbose)
        num_iters += 1
        if num_iters == max_iters:
            terminal = True
    return history


def update(i, j, v1, v2, action, track, at_start, verbose=False):
    # velocities have minimum of 0
    v1 += action[0] if v1 + action[0] >= 0 else 0
    v2 += action[1] if v2 + action[1] >= 0 else 0
    # if both velocities hit 0 at the same time, the car stalls and resets
    if v2 == v1 and v1 == 0 and at_start:
        return reset(track.shape[0] // 6 + 1)
    # Speed is capped at 5, and will the car will reset if exceeded
    max_speed = 5
    if v1 >= max_speed + 1:
        return reset(track.shape[0] // 6 + 1)
    if v2 >= max_speed:
        return reset(track.shape[0] // 6 + 1)
    # check if position(s) is safe, then update position
    traj = assign_traj(i, j, v1, v2, verbose=verbose)
    safe, terminal = is_traj_safe(traj, track)
    if not safe:
        return reset(track.shape[0] // 6 + 1)
    if terminal:
        i = (i + v1) if i + v1 < track.shape[0] else track.shape[0] - 1
        j = (j + v2) if j + v2 < track.shape[1] else track.shape[1] - 1
    else:
        i += v1
        j += v2

    return i, j, v1, v2, terminal, at_start


def reset(rows):
    return rng.choice(np.arange(rows)), 0, 0, 0, False, True


def is_traj_safe(list_of_coords, track):
    safe, terminal = True, False
    for coord in list_of_coords:
        i, j = coord
        if i >= track.shape[0] or i < 0 or j >= track.shape[1] or j < 0:
            # Wrong way/ off the track. Return as False
            safe, terminal = False, False
            return safe, terminal
        elif track[i, j] == 1:
            # Hits an obstacle. Return both as False
            safe, terminal = False, False
            return safe, terminal
        elif i == track.shape[0] - 1 and track.shape[1] > j >= 0:
            # if i is at the final row (finish line) and j hasn't gone off the rails,
            safe, terminal = True, True
            return safe, terminal

    return safe, terminal


def init_pol(states):
    return {state: tuple(rng.choice([-1, 0, 1], size=2)) for state in states}


def unif_pol(dummy):
    return tuple(rng.choice([-1, 0, 1], size=2))


def gen_states(track):
    states = []
    for i in range(0, track.shape[0]):
        for j in range(0, track.shape[1]):
            for k in range(6):
                for l in range(6):
                    if track[i, j] != 1:
                        states.append((i, j, k, l))
    return states


def test_policy():
    pass


def assign_traj(i, j, rise, run, verbose=False):
    # Traj rules divide rise/run or run/rise depending on whichever generates most divisions:
    # Then "left" justifies by clipping to int below it. 1 = 1.99 etc
    out = []
    wiggle = 0.0001  # this is to prevent X.99999 != X=1
    if rise >= run:
        i_step = 1
        j_step = (run / rise) if rise != 0 else 0
    else:
        i_step = (rise / run) if run != 0 else 0
        j_step = 1
    for k in range(np.max((rise, run))):
        i += i_step + wiggle
        j += j_step + wiggle
        out.append((int(i), int(j)))
    if verbose:
        print(out)
    return out


if __name__ == "__main__":
    # setup stuff
    track = stable_track(mode="right")
    # get state list (exhaustive)
    states = gen_states(track)
    # random starting policy
    p = init_pol(states)
    # action list consistent with states
    actions = [(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]
    pairs = []
    run_length = []
    for state in states:
        for a in actions:
            pairs.append((state, a))
    q = {(state, action): [rng.standard_normal(), 0] for state, action in
         pairs}  # 0 is number of times used for averaging
    # returns = {(state, action): [] for state, action in zip(states, actions)}

    # Begin doing runs
    num_episodes = 200
    discount = 0.75
    for i in range(num_episodes):
        run = gen_run(p, track, max_iters=1000)
        run_length.append(len(run))
        G = 0
        # print(f"Run {i+1} generated.")
        seen = []
        for step in reversed(run):
            G = discount * G + step[-1]  # -1 here returns last element in step, which is also -1 haha
            # Because this is first visit MC we only use each s,a once
            # if step in seen:
            #    continue
            # else:
            # Add s,a to seen list
            # seen.append(step)
            # Append G to returns for s , a
            # returns[(step[0], step[1])].append(G)
            # Update Q(s,a) with iterative mean
            old = q[(step[0], step[1])][0]
            q[(step[0], step[1])][1] += 1  # update count for averaging
            dnom = q[(step[0], step[1])][1]
            q[(step[0], step[1])][0] = np.around(old + 1 / dnom * (G - old), decimals=7)

            # Update policy
            best_action = np.argmax([q[step[0], a][0] for a in actions])
            p[step[0]] = (actions[best_action])

    plt.plot(np.arange(len(run_length)), run_length)
    plt.show()
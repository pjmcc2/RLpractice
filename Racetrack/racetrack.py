import numpy as np
from numpy.random import default_rng

rng = default_rng(seed=1000)


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


def gen_run(policy, track, training=True, max_iters=500):
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
    try:
        a = policy[(i, j, v1, v2)]
    except KeyError:
        a = policy(0)
    # Starting state, action
    history = [((i, j, v1, v2), a, 0)]  # remove 0?
    while not terminal:
        try:
            a = policy[(i, j, v1, v2)]
        except KeyError:
            a = policy(0)
        if training:
            if rng.uniform() <= noise_chance:
                a = (0, 0)
        history.append(((i, j, v1, v2), a, reward))
        i, j, v1, v2, terminal = update(i, j, v1, v2, a, track)
        num_iters += 1
        if num_iters == max_iters:
            terminal = True
    return history


def update(i, j, v1, v2, action, track):
    # velocities have minimum of 0
    v1 += action[0] if v1 + action[0] >= 0 else 0
    v2 += action[1] if v2 + action[1] >= 0 else 0
    # if both velocities hit 0 at the same time, the car stalls and resets
    if v2 == v1 and v1 == 0 and j != 0:
        return reset(track.shape[0])
    # Speed is capped at 5, and will the car will reset if exceeded
    max_speed = 5
    if v1 >= max_speed + 1:
        return reset(track.shape[0])
    if v2 >= max_speed:
        return reset(track.shape[0])
    # check if position is safe, then update position
    safe, terminal = is_trajectory_safe(i, j, v1, v2, track)
    if not safe:
        return reset(track.shape[0])
    if terminal:
        i = (i + v1) if i + v1 < track.shape[0] else track.shape[0] - 1
        j = (j + v2) if j + v2 < track.shape[1] else track.shape[1] - 1
    else:
        i += v1
        j += v2

    return i, j, v1, v2, terminal


def reset(rows):
    return rng.choice(rows), 0, 0, 0, False


def is_trajectory_safe(r, c, v1, v2, track):
    # TODO FIX
    if r + v1 >= track.shape[0] and c + v2 < track.shape[1]:
        terminal = True
        safe = True
    else:
        terminal = False
    if not terminal:
        if r + v1 >= track.shape[0] or c + v2 < track.shape[1]:
            safe = False
        else:
            safe = True  # no obstacles for now
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


def assign_traj(i, j, rise, run):
    # Traj rules divide rise/run or run/rise depending on whichever generates most divisions:
    # Then "left" justifies by clipping to int below it. 1 = 1.99 etc
    out = []
    if rise > run:
        i_step = 1
        j_step = run / rise
    else:
        i_step = rise / run
        j_step = 1
    for k in range(np.max((rise, run))):
        i += i_step
        j += j_step
        out.append((int(i), int(j)))

    return out


if __name__ == "__main__":
    # setup stuff
    track = stable_track(mode="blank")
    # get state list (exhaustive)
    states = gen_states(track)
    # random starting policy
    p = init_pol(states)
    # action list consistent with states
    actions = [(0, -1), (0, 0), (0, 1), (1, -1), (1, 0), (1, 1), (-1, -1), (-1, 0), (-1, 1)]
    pairs = []
    for state in states:
        for a in actions:
            pairs.append((state, a))
    q = {(state, action): [rng.standard_normal(), 0] for state, action in
         pairs}  # 0 is number of times used for averaging
    # returns = {(state, action): [] for state, action in zip(states, actions)}

    # Begin doing runs
    print(assign_traj(0,0,2,2))
    num_episodes = 0 #####
    for i in range(num_episodes):
        run = gen_run(p, track)
        G = 0
        # print(f"Run {i+1} generated.")
        seen = []
        for step in reversed(run):
            G = G + step[-1]  # -1 here returns last element in step, which is also -1 haha
            # Because this is first visit MC we only use each s,a once
            if step in seen:
                continue
            else:
                # Add s,a to seen list
                seen.append(step)
                # Append G to returns for s , a
                # returns[(step[0], step[1])].append(G)
                # Update Q(s,a) with iterative mean
                old = q[(step[0], step[1])][0]
                q[(step[0], step[1])][1] += 1  # update count for averaging
                dnom = q[(step[0], step[1])][1]
                q[(step[0], step[1])][0] = old + 1 / dnom * (step[-1] - old)

                # Update policy
                best_action = np.argmax([q[step[0], a][0] for a in actions])
                p[step[0]] = (actions[best_action])

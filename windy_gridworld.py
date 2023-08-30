import numpy as np
from numpy.random import default_rng


def move(curr_poss, action, track, mode="reg"):
    rng = default_rng()
    y, x = curr_poss
    j, i = action
    # wind comes into play:
    terminal = False
    reward = -1
    if mode == "reg":
        if x + i in [3, 4, 5, 8]:  # col numbers with wind factor 1
            j -= 1
        elif x + i in [6, 7]:  # cols with wind factor 2
            j -= 2
        else:
            j -= 0
    elif mode == "rand":
        wind_factor = rng.choice([0, 1, 2])
        j -= wind_factor
    # check if still in grid:

    if 0 <= x + i < track.shape[1] and 0 <= y + j < track.shape[0]:
        x += i
        y += j
        if track[y, x] != 0:
            terminal = True  # should reward be 0?
            reward = 0
            return (y, x), reward, terminal
        else:
            return (y, x), reward, terminal

    if x + i < 0:
        x = 0
    if y + j < 0:
        y = 0
    if x + i >= track.shape[1]:
        x = track.shape[1] - 1
    if y + j >= track.shape[0]:
        y = track.shape[0] - 1

    return (y, x), reward, terminal


# Need to do the RL stuff
def sarsa(num_episodes,step_size, epsilon, discount=1,max_iters = 250):
    #  Need policy and action-values
    gworld = init_world(7, 10)
    states = list_states(gworld)
    actions = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)] # (0,0) is possible addition
    q = init_act_vals(states, actions)
    runs = []

    for i in range(num_episodes):
        terminal = False
        s0 = (3,0)
        a0 = e_greedy(s0,q,actions,epsilon)
        num_iters = 0
        curr_run = []
        while not terminal:
            num_iters += 1
            if s0 is not None:
                s = s0
                a = a0
                s0 = None
                a0 = None
            curr_run.append((s, a))
            sPrime, r, terminal = move(s, a, gworld, mode="reg")
            aPrime = e_greedy(sPrime,q,actions,epsilon)
            q[(s,a)] = q[(s,a)] + step_size*(r +discount*q[(sPrime,aPrime)] - q[(s,a)])
            s = sPrime
            a = aPrime
            if num_iters == max_iters:
                break
        runs.append(curr_run)

    return q, runs

def init_world(r,c):
    num_rows = r
    num_columns = c
    grid = np.zeros((num_rows, num_columns))
    win_number = 42
    win_row_num = 3
    win_col_num = 7
    grid[win_row_num, win_col_num] = win_number
    return grid



def list_states(world):
    out = []
    for i in range(world.shape[0]):
        for j in range(world.shape[1]):
            out.append((i,j))

    return out

def init_act_vals(states, actions):
    combined = []
    for state in states:
        for action in actions:
            combined.append((state, action))
    return {combined[i]:0 for i in range(len(combined))}


def e_greedy(s,q,actions, epsilon,rng=None):
    rng = rng if rng is not None else default_rng()
    id = np.argmax([q[(s,a)] for a in actions])
    policy_output = actions[id]

    if rng.uniform() < epsilon:
        safety = 0
        while True:
            safety += 1
            explore = tuple(rng.choice(actions))
            if explore != policy_output:
                return explore
            elif explore == policy_output:
                explore = tuple(rng.choice(actions))
            elif safety > 1000:
                break
    else:
        return policy_output

if __name__ == "__main__":
    # Set up gridworld
    q, runs = sarsa(800, 0.1, 0.05,1)
    import matplotlib.pyplot as plt
    lengths = [len(runs[i]) for i in range(len(runs))]
    plt.plot(np.arange(len(runs)), lengths)
    plt.show()




import numpy as np
from numpy.random import default_rng
import time
import matplotlib.pyplot as plt

rng = default_rng()

import numpy as np
from numpy.random import default_rng
import time
from copy import deepcopy

rng = default_rng()


def sample(p, N):
    return np.array([p > rng.uniform() for i in range(N)])


# def init_policy():
#    return np.array([rng.integers(1, i + 1) for i in range(1, 100)])


def init_values():  # removing dummy state-value pairs of 0-0 and 100-1
    temp = rng.normal(loc=0, scale=0.1, size=99)
    return temp


def evaluate_state(state, values, action, prob, discount=1):
    # example state: capital is 50
    # Prob. of winning is 0.5:
    # Action is placing a bet of X: X=50
    # 50% chance to go to state 0, with reward 0, and 50% chance to go to state 100 w/ reward 1
    reward = 1 if state + action == 100 else 0
    sum = 0
    #next state is a loss if you go over 100 (goal is exactly 100)
    dummyG = 0 if state + action >= 100 else values[state + action-1]
    dummyL = 0 if state - action <= 0 else values[state - action-1]
    sum += prob * (reward + discount * dummyG)
    sum += (1 - prob) * (0 + discount * dummyL)
    return sum


def get_actions(state):
    return [i for i in range(1,np.min((state,100-state))+1)] # added 0 option

# chagned discount to 1
def value_iteration(values, prob, acc_min, discount=1, max_iters = 50):
    vals = values
    delta = 1
    iters = 0
    start_time = time.time()
    policy = {}
    temp_copy = deepcopy(vals)
    values_snapshots = {0: temp_copy}
    policy_snapshots = {}

    while delta > acc_min:  # 0.56298812
    #while iters < max_iters:
        delta = 0
        for i in range(1, 100):
            old_v = vals[i-1]
            # get available actions
            alist = get_actions(i)
            # Generate the list of values based on those actions
            action_value_list = [evaluate_state(i, vals, a, prob, discount) for a in alist]
            # Get the action(s) that maximize the value

            max_options = np.where(np.isclose(action_value_list, np.max(action_value_list), acc_min))
            new_vid = max_options[0][0]
            new_v = action_value_list[new_vid]
            # Save that action as new policy
            policy[i] = alist[new_vid]  # TODO check this
            # Update value
            vals[i-1] = new_v
            # Check Threshold
            delta = np.max((delta, np.abs(old_v-new_v)))
        values_snapshots[iters+1] = deepcopy(vals)
        policy_snapshots[iters] = deepcopy(policy)
        iters += 1
        if iters >= max_iters:
            print(f"Breaking after: {iters} iterations.")
            break
    print(f"Finished after {iters} iterations. Time: {time.time() - start_time}")

    return policy, values_snapshots, policy_snapshots


if __name__ == "__main__":
    v = init_values()
    policy, v_snap, p_snap = value_iteration(v, 0.5, 0.00001,max_iters=1000)

    t = [policy[i] for i in range(1, 100)]
    x = np.arange(1, 100)
    plt.step(x, t)
    plt.show()
    plt.plot(v_snap[len(v_snap)-1])
    plt.show()
    print(v_snap[len(v_snap)-1])
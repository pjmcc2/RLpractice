import numpy as np


# need to set up wind
def move(curr_poss, action, track, mode="reg"):
    x, y = curr_poss
    i, j = action
    # wind comes into play:
    if mode == "reg":
        if y + j in [3, 4, 5, 8]:  # col numbers with wind factor 1
            j += 1
        elif y + j in [6, 7]:  # cols with wind factor 2
            j += 2
        else:
            j += 0

    # check if still in grid:
    if x + i < track.shape[0]:  # is agent within the rows? if not, agent remains unchanged.
        if y + j < track.shape[1]:  # is agent within columns? If not, agent remains unchanged.
            x += i
            y += j
        else:
            return x, y
    else:
        return x, y


# Need to do the RL stuff


if __name__ == "__main__":
    # Set up gridworld
    num_rows = 7
    num_columns = 10
    grid = np.zeros(num_rows, num_columns)
    win_number = 42
    win_row_num = 3
    win_col_num = 7
    grid[win_row_num, win_col_num] = win_number



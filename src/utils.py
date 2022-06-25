import os


def makedirs(*dirnames):
    for dirname in dirnames:
        if not os.path.exists(dirname):
            os.makedirs(dirname)


def impute_observed_data(y_obs):
    observed = (y_obs == y_obs)
    y = y_obs.clone().detach()
    y[~observed] = 0.
    return observed, y
from itertools import permutations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def rotate(A, method="varimax"):
    from factor_analyzer import Rotator

    rotate = Rotator(method=method)
    rot_A = rotate.fit_transform(A)
    phi = rotate.phi_
    if phi is None:
        phi = np.identity(A.shape[1])
    # return rot_A, phi
    
    return rot_A, phi


def plot_result_df(a, b, c, d, result_path, args):
    a = a.flatten()
    b = b.flatten()
    c = c.flatten()
    d = d.flatten()
    dat = np.concatenate([a, b, c, d])

    a_label = ("proj_a" if args.a_loss == "proj" else "a")
    a_labels = np.array(["proj_a" for _ in range(len(a))])
    b_labels = np.array(["b" for _ in range(len(b))])
    c_labels = np.array(["c" for _ in range(len(c))])
    d_labels = np.array(["d" for _ in range(len(d))])
    labels = np.concatenate([a_labels, b_labels, c_labels, d_labels])

    dat = np.array([dat, labels]).T
    dat = pd.DataFrame(dat, columns=["value", "parameters"])
    dat = dat.explode("value")
    dat["value"] = dat["value"].astype("float")

    plt.cla()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    _ = sns.boxplot(data=dat, x="parameters", y="value")
    ax.set_xlabel("Parameters")
    ax.set_ylabel(args.loss)
    plt.savefig(result_path + f"/{args.n}#{args.J}#{args.K}.jpg")


def extract_config(model_name):
    model_name = model_name.split(".")[0]
    rep_times, n, J, K, _, a, factor_influ, item_depend, _ = \
        [config for config in model_name.split("#")]

    return int(rep_times), int(n), int(J), int(K), a, \
           int(factor_influ), int(item_depend)


def bias(real, pred, axis):
    return (real - pred).mean(axis=axis)


def mae(real, pred, axis):
    return np.abs(real - pred).mean(axis=axis)


def mse(real, pred, axis):
    return np.power(real - pred, 2).mean(axis=axis)


def rmse(real, pred, axis):
    return np.sqrt(mse(real, pred, axis))


def a_best_rotation(real_a, pred_a, eval_fn, method):
    j, k = real_a.shape
    rot_pred_a, rot_pred_phi = [], []
    if isinstance(pred_a, np.ndarray):
        # pred_a is a list of results from different replications. 
        pred_a = [pred_a]

    for pred_a_i in pred_a:
        _rot_pred_a, _rot_pred_phi = rotate(pred_a_i, method)
        
        # Flip the effect of A if it is negative
        _should_flip = (_rot_pred_a.mean(0) < 0)
        _rot_pred_a[:, _should_flip] *= -1
        
        # Permute the rotated matrix to find the best
        min_loss = np.inf
        for new_axis in permutations(np.arange(k)):
            pred_a_rot_permute = _rot_pred_a[:, new_axis]
            loss = eval_fn(real_a, pred_a_rot_permute, axis=None)
            if loss < min_loss:
                min_loss = loss
                rot_pred_a_i = pred_a_rot_permute
                rot_pred_phi_i = _rot_pred_phi[:, new_axis][new_axis, :]

        rot_pred_a.append(rot_pred_a_i)
        rot_pred_phi.append(rot_pred_phi_i)

    return rot_pred_a, rot_pred_phi


def a_loss(real_a, pred_a, eval_fn, axis, method):
    rot_pred_a, rot_pred_phi = a_best_rotation(real_a, pred_a, eval_fn, method)

    rot_pred_a = np.array(rot_pred_a)
    return eval_fn(real_a, rot_pred_a, axis)


def a_sparsity(real_a, pred_a, eval_fn, method):
    """
    Construct sparisity estimate of A based on CF-Quartimax Rotation. 
    Returned `a_spas` has the same type as `pred_a`.
    """
    rot_pred_a, rot_pred_phi = a_best_rotation(real_a, pred_a, eval_fn, method)
    j, k = real_a.shape
    a_spas = []
    for pred_a, pred_phi in zip(rot_pred_a, rot_pred_phi):
        psi = []
        for aj in pred_a:
            u2 = np.identity(k) + (aj * aj).sum() * pred_phi
            u = np.linalg.cholesky(u2).T
            psij = np.linalg.inv(u) @ aj
            psi.append(psij)
            
        psi = np.vstack(psi)
        if psi.shape != pred_a.shape:
            raise ValueError("Incorrect Psi matrix")
        a_spa = (np.absolute(psi) > 0.3).astype(float)
        a_spas.append(a_spa)

    if len(a_spas) == 1:
        a_spas = a_spas[0]
    return a_spas

import math

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


def sigmoid(x):
    return 1 / (1 + np.exp(-x) + 1e-8)


def is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def create_factor_cov(correlated, K, seed=1):
    np.random.seed(seed)
    if correlated:
        while True:
            factor_cov_mat = np.random.uniform(size=(K, K))
            factor_cov_mat += factor_cov_mat.T
            factor_cov_mat /= 2
            np.fill_diagonal(factor_cov_mat, 1.)
            if is_pos_def(factor_cov_mat):
                break
    else:
        factor_cov_mat = np.identity(K)
    return factor_cov_mat


class MirtDataset(Dataset):
    def __init__(self, n, j, k, pl, seed=1, **kwargs):
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        if "device" in kwargs:
            self.device = kwargs["device"]
        else:
            self.device = "cpu"

        X = self._generate_x(n, k, **kwargs)
        a = self._generate_a(j, k, **kwargs)
        b = self._generate_b(j)
        c, d = self._generate_cd(j, pl, **kwargs)

        link = kwargs.get("link", sigmoid)
        logit = b + X @ a.T
        prob = c + (d - c) * link(logit)
        Y_full = np.random.binomial(n=1, p=prob)
        Y_observed_indi = self._generate_y_observed_mask(n, j, **kwargs)
        Y = np.full((n, j), fill_value=np.nan)
        Y[Y_observed_indi] = Y_full[Y_observed_indi]

        self.X = torch.from_numpy(X).to(self.device)
        self.a = torch.from_numpy(a).to(self.device)
        self.b = torch.from_numpy(b).to(self.device)
        self.c = torch.from_numpy(c).to(self.device)
        self.d = torch.from_numpy(d).to(self.device)
        self.Y_full = torch.from_numpy(Y_full).to(self.device)
        self.Y = torch.from_numpy(Y).float().to(self.device)

    def __len__(self):
        return self.Y.shape[0]

    def __getitem__(self, item):
        return self.Y[item]

    def _generate_x(self, n, k, **kwargs):
        factor_cov_mat = kwargs.get("factor_cov_mat", np.identity(k))
        return np.random.multivariate_normal(
            mean=np.zeros(k), cov=factor_cov_mat, size=n)

    def _generate_a(self, j, k, **kwargs):
        """
        `factor_influ` is an auxiliary parameter, which determines step lengths 
        in rows of diagonal or s-shaped `a`.
        For an `a` matrix with `factor_influ` = 10, its first ten
        rows are non-zero in the first `item_depend` columns.
        """
        dist = kwargs.get("a_dist", "uniform")
        shape = kwargs.get("a_shape", "diag")
        factor_influ = kwargs.get("factor_influ", 1)
        item_depend = kwargs.get("item_depend", 1)

        assert shape in ["diag", "s", "upper", "lower"], "invalid `shape` value"
        assert factor_influ > 0, "invalid `factor_influ` value"
        assert item_depend > 0, "invalid `item_depend` value"

        if shape in ["upper", "lower"]:
            if dist == "uniform":
                a = np.random.uniform(size=(j, k), low=0.5, high=1.5)
            else:
                a = np.random.lognormal(size=(j, k))
            if shape == "upper":
                return np.triu(a)
            else:
                return np.tril(a)

        rep_j = (k - item_depend + 1) * factor_influ
        rep_block = math.ceil(j / rep_j)

        if dist == "uniform":
            a_block_full = np.random.uniform(size=(rep_j, k), low=0.5, high=1.5)
        else:
            a_block_full = np.random.lognormal(size=(rep_j, k))

        a_block = np.zeros(shape=(rep_j, k))
        for col in range(k):
            if (col + item_depend - 1) > k:
                break
            row = col * factor_influ
            a_block[row:(row + factor_influ), col: (col + item_depend)] = \
                a_block_full[row:(row + factor_influ), col: (col + item_depend)]
        ext_a = []
        for _ in range(rep_block):
            ext_a.append(a_block)
            if shape == "s":
                a_block = np.flip(a_block, 0)
        ext_a = np.concatenate(ext_a, axis=0)
        return ext_a[:j, ]

    def _generate_b(self, j, **kwargs):
        with_b = bool(kwargs.get("with_b", True))
        return np.random.normal(size=(j,)) if with_b else np.zeros(shape=(j,))

    def _generate_cd(self, j, pl, **kwargs):
        constant_cd = bool(kwargs.get("constant_cd", False))

        if constant_cd:
            # A simple version of c = 0.1 and d = 0.9
            c = np.ones(shape=(j,)) * 0.1 if pl > 2 else np.zeros(shape=(j,))
            d = np.ones(shape=(j,)) * 0.9 if pl > 3 else np.ones(shape=(j,))
        else:
            c = np.random.beta(
                size=(j,), a=1, b=9) if pl > 2 else np.zeros(shape=(j,))
            if pl > 3:
                d = np.zeros(shape=(j,))
                # d must be as large as c
                while (d - c).min() < 0.:
                    d = np.random.beta(size=(j,), a=9, b=1)
            else:
                d = np.ones(shape=(j,))

        return c, d

    def _generate_y_observed_mask(self, n, j, **kwargs):
        max_observed = kwargs.get("max_observed", j)
        shuffle = bool(kwargs.get("shuffle_before_mask", False))
        # We create a block diagonal
        block_columns = j // max_observed
        block_rows = n // block_columns
        mask = np.zeros((n, j))
        for diag in range(block_columns):
            mask[(diag * block_rows): ((diag + 1) * block_rows),
            (diag * max_observed):((diag + 1) * max_observed)] = 1
        if shuffle:
            rows = np.arange(n)
            np.random.shuffle(rows)
            mask = mask[rows]
        return (mask == 1)

    def plot_a(self, figsize=10):
        fig = plt.figure(figsize=(figsize, 5 * figsize))
        ax = fig.add_subplot(111)
        # plt.matshow(sig_var_all, cmap=plt.cm.Blues)
        a = self.a.numpy()
        j, k = a.shape
        caxes = ax.matshow(a, cmap=plt.cm.Blues, interpolation="none")
        fig.colorbar(caxes)

        ax.set_xticks(np.arange(1, k + 1))
        ax.set_xticklabels(range(1, k + 1))
        ax.tick_params(axis="x", bottom=True, top=False,
                       labelbottom=True, labeltop=False)

        ax.set_yticks(np.arange(1, j + 1))
        ax.set_yticklabels(range(1, j + 1))
        ax.tick_params(axis="y", bottom=True, top=False,
                       labelbottom=True, labeltop=False)

        plt.xlabel("Latent factors")
        plt.ylabel("Items")


def load_data(n, j, k, pl, link=sigmoid, with_b=True, seed=1,
              a_dist="uniform", a_shape="s", factor_influ=5, item_depend=1,
              max_observed=None, correlated_factor=False,
              svd_init=False):

    seed = int(seed)
    J = int(j)
    K = int(k)
    n = int(n)
    pl = int(pl)

    factor_influ = int(factor_influ)
    a_dist = str(a_dist)
    a_shape = str(a_shape)
    item_depend = int(item_depend)

    if not max_observed:
        max_observed = J
    max_observed = int(max_observed)

    svd_init = bool(svd_init)

    factor_cov_mat = create_factor_cov(correlated_factor, K, seed)
    data = MirtDataset(n, J, K, pl, link=link, with_b=with_b, seed=seed,
                       factor_cov_mat=factor_cov_mat,
                       a_dist=a_dist, a_shape=a_shape,
                       factor_influ=factor_influ,
                       item_depend=item_depend,
                       max_observed=max_observed)

    Y = data.Y.numpy()
    X = data.X.numpy()
    a = data.a.numpy()
    b = data.b.numpy()
    c = data.c.numpy()
    d = data.d.numpy()

    if svd_init:
        raise NotImplementedError("Due to module import issue, "
                                  "SVD initialization is not supported.")
    return Y, X, a, b, c, d
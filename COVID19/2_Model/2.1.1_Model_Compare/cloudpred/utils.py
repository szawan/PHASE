import cloudpred
import time
import scipy
import scipy.io
import os
import numpy as np
import pandas as pd
import glob
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import pickle
import sklearn.metrics
import math
import copy
import datetime
import pathlib
import tqdm
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_counts(filename):
    counts = scipy.sparse.load_npz(filename)
    counts = counts.astype(np.float64)
    sample_info = os.path.splitext(os.path.basename(filename))[0]
    ct_filename = os.path.join(os.path.dirname(filename), "ct_" + os.path.splitext(os.path.basename(filename))[0] + ".npy")
    if os.path.isfile(ct_filename):
        ct = np.load(ct_filename, allow_pickle=True)
    else:
        ct = None

    return counts.transpose(), ct, sample_info

def load_synthetic(root, valid=0.25, test=0.25, train_patients=None, cells=None):
    try: 
        with open(root + "/Xall.pkl", "rb") as f:
            Xall = pickle.load(f)
        with open(root + "/state.pkl", "rb") as f:
            state = pickle.load(f)
    except FileNotFoundError:
        Xall = []
        state = []
        for dirname in sorted(glob.iglob(root + "/*")): 
            if os.path.isdir(dirname) and os.path.basename(dirname) != "Test":
                if os.path.basename(dirname) == "regression":
                    X = []
                    for filename in tqdm.tqdm(sorted(glob.iglob(dirname + "/*.npz"))):
                        x = load_counts(filename)
                        value_filename = os.path.join(os.path.dirname(filename), "value_" + os.path.splitext(os.path.basename(filename))[0] + ".npy")
                        y = np.load(value_filename).item()
                        X.append((x[0], y, x[1]))
                    state.append(os.path.basename(dirname))
                    Xall.append(X)
                else: 
                    X = []
                    for filename in tqdm.tqdm(sorted(glob.iglob(dirname + "/*.npz"))):
                        X.append(load_counts(filename))
                    X = list(map(lambda x: (x[0], len(state), x[1], x[2]), X))

                    state.append(os.path.basename(dirname))
                    Xall.append(X)


        with open(root + "/Xall.pkl", "wb") as f:  
            pickle.dump(Xall, f)
        with open(root + "/state.pkl", "wb") as f: 
            pickle.dump(state, f)

    for X in Xall:
        random.shuffle(X) 

    if isinstance(valid, float) and isinstance(test, float): 
        Xvalid = [y for x in Xall for y in x[:round(len(x) * valid)]] 
        Xtest = [y for x in Xall for y in x[round(len(x) * valid):(round(len(x) * valid) + round(len(x) * test))]]
        Xtrain = [y for x in Xall for y in x[(round(len(x) * valid) + round(len(x) * test)):]]
        
    elif isinstance(valid, int) and isinstance(test, int): 
        Xvalid = [y for x in Xall for y in x[:valid]]
        Xtest = [y for x in Xall for y in x[valid:(valid + test)]]
        Xtrain = [y for x in Xall for y in x[(valid + test):]]
        
    else:
        raise TypeError()

    random.shuffle(Xtrain) 
    random.shuffle(Xvalid)
    random.shuffle(Xtest)
 

    if train_patients is not None:
        Xtrain = Xtrain[:train_patients]

    if cells is not None:
        for i in range(len(Xtrain)):
            if Xtrain[i][0].shape[0] > cells:
                Xtrain[i] = list(Xtrain[i])
                ind = np.random.choice(Xtrain[i][0].shape[0], cells, replace=False)
                Xtrain[i][0] = Xtrain[i][0][ind, :]
                Xtrain[i][2] = Xtrain[i][2][ind]
                Xtrain[i] = tuple(Xtrain[i])
        for i in range(len(Xvalid)):
            if Xvalid[i][0].shape[0] > cells:
                Xvalid[i] = list(Xvalid[i])
                ind = np.random.choice(Xvalid[i][0].shape[0], cells, replace=False)
                Xvalid[i][0] = Xvalid[i][0][ind, :]
                Xvalid[i][2] = Xvalid[i][2][ind]
                Xvalid[i] = tuple(Xvalid[i])
        for i in range(len(Xtest)):
            if Xtest[i][0].shape[0] > cells:
                Xtest[i] = list(Xtest[i])
                ind = np.random.choice(Xtest[i][0].shape[0], cells, replace=False)
                Xtest[i][0] = Xtest[i][0][ind, :]
                Xtest[i][2] = Xtest[i][2][ind]
                Xtest[i] = tuple(Xtest[i])
        
    return Xtrain, Xvalid, Xtest, state
    

population = [
                'NK_16hi', 'B_switched_memory', 'B_naive', 'B_immature', 'DC2',
                'HSC_CD38pos', 'CD4.Tfh', 'Plasmablast', 'CD14_mono', 'Platelets',
                'CD4.IL22', 'CD8.EM', 'CD83_CD14_mono', 'CD4.Naive', 'pDC',
                'NK_prolif', 'Plasma_cell_IgA', 'C1_CD16_mono', 'Plasma_cell_IgM',
                'CD16_mono', 'DC3', 'Plasma_cell_IgG', 'NK_56hi', 'RBC',
                'CD8.Naive', 'CD4.CM', 'MAIT', 'NKT', 'HSC_CD38neg', 'HSC_prolif',
                'CD8.TE', 'CD4.Prolif', 'CD4.EM', 'B_exhausted', 'DC1',
                'CD8.Prolif', 'HSC_erythroid', 'gdT', 'B_non-switched_memory',
                'ASDC', 'Mono_prolif', 'CD4.Th1', 'ILC2', 'DC_prolif', 'Treg',
                'ILC1_3', 'CD4.Th2', 'B_malignant', 'HSC_MK', 'HSC_myeloid','CD4.Th17'
             ]


def load_purified_population(test_frac=0.1):

    Xtrain = []
    Ytrain = []
    Xtest = []
    Ytest = []

    for (i, p) in enumerate(population):
        print("p: ",p)
        x = load_counts("data/MatFiles/" + p + ".mat")

        index = np.arange(x.shape[0])
        np.random.shuffle(index)
        x = x[index, :]

        y = np.zeros(x.shape[0], np.int64)
        y[:] = i

        n = round(x.shape[0] * (1 - test_frac))
        Xtrain.append(x[:n, :])
        Ytrain.append(y[:n])
        Xtest.append (x[n:, :])
        Ytest.append (y[n:])

    Xtrain = scipy.sparse.vstack(Xtrain)
    Xtest  = scipy.sparse.vstack(Xtest)
    Ytrain = np.concatenate(Ytrain)
    Ytest  = np.concatenate(Ytest)

    index = np.arange(Xtrain.shape[0])
    np.random.shuffle(index)
    Xtrain = Xtrain[index, :]
    Ytrain = Ytrain[index]

    index = np.arange(Xtest.shape[0])
    np.random.shuffle(index)
    Xtest = Xtest[index, :]
    Ytest = Ytest[index]

    return Xtrain, Ytrain, Xtest, Ytest, len(population)


def scipy_sparse_to_pytorch(x):
    x = x.tocoo()
    v = torch.Tensor(x.data)
    i = torch.LongTensor([x.row, x.col])
    return torch.sparse.FloatTensor(i, v, x.shape)


# https://stackoverflow.com/questions/4623446/how-do-you-sort-files-numerically
def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def loglevel(level):
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    return numeric_level


def str2bool(v):
    return v.lower() in ("yes", "y", "true", "t", "1")


# Based on https://mail.python.org/pipermail/python-list/2010-November/591474.html
class MultilineFormatter(logging.Formatter):
    def format(self, record):
        str = logging.Formatter.format(self, record)
        header, footer = str.split(record.message)
        str = str.replace('\n', '\n' + ' '*len(header))
        return str


def split(X, y, train=0.5, valid=0.25, test=0.25):
    assert(abs(train + valid + test - 1) < 1e-5)

    n = len(X)
    index = np.arange(n)
    np.random.shuffle(index)

    X = [X[i] for i in index]
    y = [y[i] for i in index]

    n1 = round(n * train)
    n2 = round(n * valid) + n1

    return X[:n1], y[:n1], X[n1:n2], y[n1:n2],  X[n2:], y[n2:],


def latexify():
    import matplotlib
    params = {'backend': 'pdf',
              'axes.labelsize':  8,
              'font.size':       8,
              'legend.fontsize': 8,
              'xtick.labelsize': 8,
              'ytick.labelsize': 8,
              'font.family': 'DejaVu Serif',
              'font.serif': 'Computer Modern',
             }
    matplotlib.rcParams.update(params)


def train_classifier(Xtrain, Xvalid, Xtest, classifier, regularize=None,
                     eta=1e-8, iterations=3, stochastic=True, cuda=False, state=None, regression=False):

    logger = logging.getLogger(__name__)

    if torch.cuda.is_available() and cuda:
        classifier.cuda()

    if regression:
        criterion = torch.nn.modules.MSELoss()
    else:
        criterion = torch.nn.modules.CrossEntropyLoss()


    batch = {"train": [None for _ in range(len(Xtrain))],
             "valid": [None for _ in range(len(Xvalid))],
             "test": [None for _ in range(len(Xtest))]}

    optimizer = torch.optim.SGD(classifier.parameters(), lr=eta, momentum=0.9)

    log = []
    best_res = {"accuracy": -float("inf")}
    best_res = {"loss": float("inf")}
    best_model = None
    for iteration in range(iterations):
        logger.debug("Iteration #" + str(iteration + 1) + ":")
        # logger.debug(list(classifier.parameters()))

        for dataset in ["train", "valid"] + (["test"] if (iteration == (iterations - 1)) else []):

            if dataset == "train":
                logger.debug("    Training:")
                X = Xtrain
            elif dataset == "valid":
                logger.debug("    Validation:")
                X = Xvalid
            elif dataset == "test":
                logger.debug("    Testing:")
                X = Xtest
            else:
                raise NotImplementedError()
            n = len(X)

            total = 0.
            correct = 0
            prob = 0.
            loss = 0.
            y_score = []
            y_true = []
            y_prob = []
            reg = 0.
            
            for (start, (x, y, *_)) in enumerate(X):
                if batch[dataset][start] is None:
                    if isinstance(x, torch.Tensor):
                        pass
                    elif isinstance(x, np.ndarray):
                        x = torch.Tensor(x)
                    else:
                        x = x.tocoo()
                        v = torch.Tensor(x.data)
                        i = torch.LongTensor([x.row, x.col])
                        x = torch.sparse.FloatTensor(i, v, x.shape)

                    if dataset != "test":
                        if regression:
                            y = torch.FloatTensor([y])
                        else:
                            y = torch.LongTensor([y])

                    if torch.cuda.is_available() and cuda:
                        x = x.cuda()
                        if dataset != "test":
                            y = y.cuda()

                    batch[dataset][start] = (x, y)
                else:
                    x, y = batch[dataset][start]

                t = time.time()
                classifier = classifier.to(device)
                z = classifier(x)


                if regression:
                    if len(z.shape) == 2:
                        if z.shape[1] == 1:
                            z = z[:, 0]
                        elif z.shape[1] == 2:
                            z = z[:, 1]
                    y_score.append(z[0].detach().numpy().item())
                else:
                    probabilities = F.softmax(z, dim=1)
                    y_score.append(torch.argmax(probabilities, dim=1).item())


                
                y_true.append(y.detach().numpy().item())
                pred = torch.argmax(z) 
                y_prob.append(probabilities.detach().cpu().numpy().tolist())
                y_prob_flat = np.array(y_prob).reshape(len(y_prob), -1)
 
                
                if dataset != "test":
                    if not regression:
                        probabilities = F.softmax(z, dim=1)
                        prob += probabilities[0, y].detach().cpu().numpy()
                        # print("prob: ", prob)
                        correct += torch.sum(pred.to(device) == y.to(device)).item()
                        # correct += torch.sum(pred == y).cpu().numpy()
                        # print("correct: ", correct)
                        z = z.to(device)
                        y = y.to(device)
                        l = criterion(z, y)
                        # print("loss: ",l)                      
                    if stochastic:
                        loss = l
                    else:
                        loss += l
                    total += l.detach().cpu().numpy()
                    # print("total: ",total)
                    if regularize is not None:
                        r = regularize(classifier)
                        loss += r
                        reg += r.detach().cpu().numpy()

                if dataset == "train" and (stochastic or start + 1 == len(X)):
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                elif dataset == "test":
                    pred = torch.argmax(z, dim=1).numpy()
                    # if len(pred.shape) != 0:
                    #     pred = pred[0]
                    if state is not None:
                        pred = state[pred.item()]
                    logger.info(y + "\t" + str(pred))

            if dataset != "test" and n != 0:
                res = {}
                res["loss"] = total / float(n)
                res["accuracy"] = correct / float(n)
                res["soft"] = prob / float(n)
                y_true_onehot = label_binarize(y_true, classes=np.unique(y_true)).reshape(-1, len(np.unique(y_true)))

                if any(map(math.isnan, y_score)):
                    res["auc"] = float("nan")
                    # res["r2"] = float("nan")
                elif regression:
                    res["auc"] = float("nan")
                    res["r2"] = sklearn.metrics.r2_score(y_true, y_score)
                else:
                    
                    res["auc"] = sklearn.metrics.roc_auc_score(y_true_onehot, y_prob_flat, multi_class='ovr')
                    res["precision_score"] = sklearn.metrics.precision_score(y_true, y_score, average='weighted', zero_division=1)
                    res["recall_score"] = sklearn.metrics.recall_score(y_true, y_score, average='weighted')
                    res["F1_score"] = sklearn.metrics.f1_score(y_true, y_score, average='weighted')
                    

                logger = logging.getLogger(__name__)
                logger.debug("        Loss:            " + str(res["loss"]))
                logger.debug("        Accuracy:        " + str(res["accuracy"]))
                logger.debug("        AUC:             " + str(res["auc"]))
                logger.debug("        Precision_score: " + str(res["precision_score"]))
                logger.debug("        recall_score:    " + str(res["recall_score"]))
                logger.debug("        F1_score:        " + str(res["F1_score"]))
                
                

                if regularize is not None:
                    logger.debug("        Regularize:    " + str(reg / float(n)))
            if dataset == "train":
                log.append([])
            log[-1].append((total / float(n), correct / float(n)) if n != 0 else (None, None))
            if dataset == "valid":
                if res["loss"] <= best_res["loss"]:
                    best_res = res
                    best_model = copy.deepcopy(classifier.state_dict())
    classifier.load_state_dict(best_model)
    return classifier, best_res


def calibrate_smoothing(X, density,
                     eta=1e-4, iterations=1):
    # TODO: 
    X = torch.Tensor(X)
    X = X[torch.randperm(X.shape[0]), :]

    optimizer = torch.optim.SGD(density.parameters(), lr=eta, momentum=0.9)
    for iteration in range(iterations):
        print("Iteration #" + str(iteration + 1) + ":")
        print(X.shape)
        total = 0.
        for start in range(X.shape[0]):

            mu = X[start, :]
            if start == 0:
                x = X[(start + 1):, :]
            elif start + 1 == X.shape[0]:
                x = X[:start, :]
            else:
                x = torch.cat([X[:start, :], X[(start + 1):, :]])
            x = X[torch.randperm(X.shape[0])[:1000], :]

            if start % 100 == 0:
                print(start)
                print(list(density.parameters()))
            d = density(mu, x)
            d = -torch.log(torch.mean(d))
            optimizer.zero_grad()
            d.backward()
            optimizer.step()

    return density


def logsumexp(x, dim=None):
    if dim is None:
        m = torch.max(x)
        return m + torch.log(torch.sum(torch.exp(x - m)))
    else:
        m, _ = torch.max(x, dim, keepdim=True)
        return m + torch.log(torch.sum(torch.exp(x - m), dim))


def logmeanexp(x, dim=None):
    if dim is None:
        return logsumexp(x) - math.log(np.prod(x.shape))
    else:
        return logsumexp(x, dim) - math.log(x.shape[dim])

def setup_logging(logfile=None, loglevel=logging.DEBUG):
    if logfile is None:
        logfile = "log/" + datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    pathlib.Path(os.path.dirname(logfile)).mkdir(parents=True, exist_ok=True)

    cfg = dict(
          version=1,
          formatters={
              "f": {"()":
                        "cloudpred.utils.MultilineFormatter",
                    "format":
                        "%(levelname)-8s [%(asctime)s] %(message)s",
                    "datefmt":
                        "%m/%d %H:%M:%S"}
              },
          handlers={
              "s": {"class": "logging.StreamHandler",
                    "formatter": "f",
                    "level": loglevel},
              "f": {"class": "logging.FileHandler",
                    "formatter": "f",
                    "level": logging.DEBUG,
                    "filename": logfile}
              },
          root={
              "handlers": ["s", "f"],
              "level": logging.NOTSET
              },
      )
    logging.config.dictConfig(cfg)


class Aggregator(torch.nn.Module):
    def __init__(self):
        super(Aggregator, self).__init__()

    def forward(self, x):
        return torch.mean(x, dim=0).unsqueeze_(0)



def train_pca_autoencoder(Xtrain, Ytrain, Xtest, Ytest, dims, transform,
                          iterations=25, batch_size=1024, figroot=None):
    # TODO: sparsify?
    logger = logging.getLogger(__name__)

    subsample = 10000
    Xtrain = scipy.sparse.csr_matrix(Xtrain)
    Xtest = scipy.sparse.csr_matrix(Xtest)
    if Xtrain.shape[0] > subsample:
        ind = np.random.choice(Xtrain.shape[0], subsample, replace=False)
        Xtrain = Xtrain[ind, :]
    if Xtest.shape[0] > subsample:
        ind = np.random.choice(Xtest.shape[0], subsample, replace=False)
        Xtest = Xtest[ind, :]

    mu = np.mean(Xtrain, 0)
    pc = np.random.randn(Xtrain.shape[1], dims)
    pc /= np.linalg.norm(pc, axis=0, keepdims=True)

    if Xtest is not None:
        total = 0.
        for start in range(0, Xtest.shape[0], batch_size):
            end = min(start + batch_size, Xtest.shape[0])
            x = Xtest[start:end, :].toarray()
            total += np.linalg.norm(x - mu) ** 2
        logger.debug("MSE: " + str(total / Xtest.size))

    for iteration in range(iterations):
        t = time.time()
        logger.debug("Iteration #" + str(iteration + 1) + ":")
  
        new = 0 * pc
        for start in range(0, Xtrain.shape[0], batch_size):
            end = min(start + batch_size, Xtrain.shape[0])
            x = Xtrain[start:end, :].toarray() - mu
            new += np.matmul(x.transpose(), np.matmul(x, pc)) / (end - start)
        logger.debug("    " + str(np.linalg.norm(new, axis=0)))
        pc, _ = np.linalg.qr(new)

        if Xtest is not None:
            total = 0.
            for start in range(0, Xtest.shape[0], batch_size):
                end = min(start + batch_size, Xtest.shape[0])
                x = Xtest[start:end, :].toarray()
                latent = np.matmul(x - mu, pc)
                xhat = np.matmul(latent, pc.transpose())

                total += np.linalg.norm(x - mu - xhat) ** 2
            logger.debug("    MSE: " + str(total / Xtest.size))
        logger.debug("    Time: " + str(time.time() - t))

    return pc

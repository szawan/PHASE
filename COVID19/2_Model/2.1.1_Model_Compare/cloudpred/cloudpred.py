import copy
import cloudpred
import numpy as np
import sklearn.mixture
import torch
import math
import logging

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train(Xtrain, Xvalid, centers=2, regression=False): 
    X = np.concatenate([x.cpu().numpy() for (x, *_) in Xtrain]) 
    gm = []
    for X, *_ in Xtrain:
        gm.append(X)
        
    gm = np.concatenate([g.cpu().numpy() for g in gm])    
    model = sklearn.mixture.GaussianMixture(n_components=centers, covariance_type = "diag") 
    gm = model.fit(gm) 

    component = [Gaussian(torch.Tensor(gm.means_[i, :]),
                          torch.Tensor(1. / gm.covariances_[i, :])) for i in range(centers)] 
    mixture = Mixture(component, gm.weights_) 
    classifier = DensityClassifier(mixture, centers, 3).to(device) 

    X = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach() for (X, y, *_) in Xtrain]).to(device) 
    if regression:
        y = torch.FloatTensor([y for (X, y, *_) in Xtrain]).to(device) 
    else:
        y = torch.LongTensor([y for (X, y, *_) in Xtrain]).to(device) 
        
    
    Xv = torch.cat([mixture(torch.Tensor(X)).unsqueeze_(0).detach()for (X, y, *_) in Xvalid]).to(device)
    if regression:
        yv = torch.FloatTensor([y for (X, y, *_) in Xvalid]).to(device)
    else:
        yv = torch.LongTensor([y for (X, y, *_) in Xvalid]).to(device)

    logger = logging.getLogger(__name__)
  
    for lr in [1e-2, 1e-3,1e-4]:
        optimizer = torch.optim.SGD(classifier.pl.parameters(), lr=lr, momentum=0.9) 
        if regression:
            criterion = torch.nn.modules.MSELoss() 
        else:
            criterion = torch.nn.modules.CrossEntropyLoss() 
        best_loss = float("inf")
        best_model = copy.deepcopy(classifier.pl.state_dict()) 
        logger.debug("Learning rate: " + str(lr)) 
     
        for i in range(1000):
            z = classifier.pl(X) 
    
            if regression:
                z = z[:, 1]
            loss = criterion(z, y) 

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            zv = classifier.pl(Xv) 
            if regression:
                zv = zv[:, 1]
            loss = criterion(zv, yv)
            if i % 100 == 0:
                logger.debug(str(loss.detach().cpu().numpy()))
            if loss < best_loss:
                best_loss = loss
                best_model = copy.deepcopy(classifier.pl.state_dict()) 
        classifier.pl.load_state_dict(best_model) 

    
    reg = None
    return cloudpred.utils.train_classifier(Xtrain, Xvalid, [], classifier, regularize=reg,
                                            iterations=1000, eta=1e-4, stochastic=True,
                                            regression=regression)


def eval(model, Xtest, regression=False):
    reg = None
    model, res = cloudpred.utils.train_classifier([], Xtest, [], model, regularize=reg,
                                                  iterations=1, eta=0, stochastic=True,
                                                  regression=regression)
    return res


class Gaussian(torch.nn.Module):
    def __init__(self, mu, invvar):
        super(Gaussian, self).__init__()
        self.mu = torch.nn.parameter.Parameter(torch.Tensor(mu),requires_grad=True)
        self.invvar = torch.nn.parameter.Parameter(torch.Tensor(invvar),requires_grad=True)


    def forward(self, x): 
        invvar = torch.abs(self.invvar).clamp(1e-5)
        return -0.5 * (math.log(2 * math.pi) - torch.sum(torch.log(invvar))
                       + torch.sum((self.mu.to(device) - x.to(device)) ** 2 * invvar, dim=1))


class Mixture(torch.nn.Module):
    def __init__(self, component, weights):
        super(Mixture, self).__init__()
        self.component = torch.nn.ModuleList(component)
        self.weights = torch.nn.parameter.Parameter(torch.Tensor(weights).unsqueeze_(1))

    def forward(self, x):
        logp = torch.cat([c(x).unsqueeze(0) for c in self.component])
        shift, _ = torch.max(logp, 0)
        p = torch.exp(logp - shift) * self.weights
        return torch.mean(p / torch.sum(p, 0), 1)

class DensityClassifier(torch.nn.Module):
    def __init__(self, mixture, centers, states=2):
        super(DensityClassifier, self).__init__()
        self.mixture = mixture
        self.pl = PolynomialLayer(centers, states)

    def forward(self, x):     
        self.d = self.mixture(x).unsqueeze_(0)
        return self.pl(self.d)


class PolynomialLayer(torch.nn.Module):
    def __init__(self, centers, states=2): 
        super(PolynomialLayer, self).__init__()
        self.polynomial = torch.nn.ModuleList([Polynomial(centers) for _ in range(states - 1)]) 

    def forward(self, x): 
        return torch.cat([torch.zeros(x.shape[0], 1).to(x.device)]
                         + [p(x).unsqueeze_(1) for p in self.polynomial], dim=1)


class Polynomial(torch.nn.Module):
    def __init__(self, centers=1, degree=2): 
        super(Polynomial, self).__init__()
        self.centers = centers
        self.degree = degree
        self.a = torch.nn.parameter.Parameter(torch.zeros(degree, centers))
        self.c = torch.nn.parameter.Parameter(torch.zeros(1))

    def forward(self, x): 
        return torch.sum(sum([self.a[i, :] * (x ** (i + 1)) for i in range(self.degree)]), dim=1) + self.c

    def linear_reg(self, xy): 
        x = np.concatenate(list(map(lambda x: x[0].reshape(1, -1), xy)))
        y = np.array(list(map(lambda x: x[1], xy)))
        y = 2 * y - 1
        x = np.concatenate([x ** (i + 1) for i in range(self.degree)] + [np.ones((x.shape[0], 1))], axis=1)
        w = np.dot(np.linalg.pinv(x), y)
        self.a.data = torch.Tensor(w[:-1].reshape(self.degree, self.centers))
        self.c.data = torch.Tensor([w[-1]])


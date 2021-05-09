# Vectorized version of tsne
# much faster for smaller n (n < 1000)
# could develop further for batching
# Senyu Tong
# Based on Xiao Li's implmentation,
# requires pytorch 1.8

import numpy as np
import argparse
import torch
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="wiki40b_150k_reps.dat", help="file name of feature stored")
parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")
parser.add_argument("--input_dim", type=int, default=128, help="input dimension")
parser.add_argument("--output_dim", type=int, default=64, help="output dimension")
parser.add_argument("--output_path", type=str, default="wiki40b_150k_reduced.dat", help="file name of output path")

opt = parser.parse_args()
print("get choice from args", opt)
data = opt.data


torch.set_default_tensor_type(torch.cuda.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)

    sumP = torch.sum(P)

    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP

    return H, P


def x2p_torch(X, tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape

    sum_X = torch.sum(X*X, 1)
    D = torch.add(torch.add(-2 * torch.mm(X, X.t()), sum_X).t(), sum_X)

    P = torch.zeros(n, n)
    beta = torch.ones(n, 1)
    logU = torch.log(torch.tensor([perplexity]))
    n_list = [i for i in range(n)]

    # Loop over all datapoints
    for i in tqdm(range(n)):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        # there may be something wrong with this setting None
        betamin = None
        betamax = None
        Di = D[i, n_list[0:i]+n_list[i+1:n]]

        (H, thisP) = Hbeta_torch(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while torch.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].clone()
                if betamax is None:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].clone()
                if betamin is None:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta_torch(Di, beta[i])

            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, n_list[0:i]+n_list[i+1:n]] = thisP

    # Return final P-matrix
    return P


def pca_torch(X, no_dims=64):
    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - torch.mean(X, 0)

    (l, M) = torch.eig(torch.mm(X.t(), X), True)
    # split M real
    for i in range(d):
        if l[i, 1] != 0:
            M[:, i+1] = M[:, i]
            i += 1

    Y = torch.mm(X, M[:, 0:no_dims])
    return Y


def tsne(X, no_dims=64, perplexity=30.0, pca_pre=False):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Initialize variables
    if pca_pre:
        X = pca_torch(X, 64)

    (n, d) = X.shape
    max_iter = 1000
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = torch.randn(n, no_dims)
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    P = x2p_torch(X, 1e-5, perplexity)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * 4.    # early exaggeration
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in tqdm(range(max_iter)):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q
        '''
        # two for loops, naive implementation
        T = PQ.T[:n] * num
        for i in range(n):
            for j in range(5):
                dY[i][j] = 0
                for k in range(n):
                    dY[i][j] +=  T[i][k] * (Y[i][j] - Y[k][j])

        # 1 for loop, vectorized
        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)
        '''
        # Much faster for smaller n (n < 1000) fully-vectorized version
        A = Y.unsqueeze(1).repeat(1, n, 1).view(-1, no_dims)
        B = Y.tile((n, 1))
        C = ((PQ.T[:n] * num).reshape(-1, 1).repeat(1, no_dims).view(n * n,no_dims) * (A - B))
        dY = torch.sum(C.view(n, n, no_dims), 1)


        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y


if __name__ == "__main__":
    X = np.memmap(opt.data, dtype='float32', mode='r', shape=(150285, 128))
    X = torch.Tensor(X)

    '''
    X = torch.randn((5000, 128))
    print(X.shape)
    '''

    with torch.no_grad():
        Y = tsne(X, 64, 128, 20.0)
    print(Y.shape)


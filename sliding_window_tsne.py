# sliding-sindow version of tsne, vectorized for small n
# Senyu Tong
# Based on Xiao Li's implmentation,
# requires pytorch 1.8

import numpy as np
import argparse
import torch
from tqdm import tqdm


torch.set_default_tensor_type(torch.cuda.DoubleTensor)


def Hbeta_torch(D, beta=1.0):
    P = torch.exp(-D.clone() * beta)
    sumP = torch.sum(P)
    H = torch.log(sumP) + beta * torch.sum(D * P) / sumP
    P = P / sumP
    torch.cuda.empty_cache()
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
    for i in tqdm(range(n), position=0, leave=True):

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

    torch.cuda.empty_cache()
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


# init_method: 0: pca, 1: random, 2: prev_epoch
def tsne(X, window_size=5000, jump_size=1000, prev_feat=None, no_dims=64, perplexity=30.0, init_method="pca", max_iter=1000, initial_momentum=0.5, final_momentum=0.8, eta=500, min_gain=0.01, tol=1e-5, initial_iter=20, early_exag=100, exag_factor=4.):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """

    # Initialize variables
    if prev_feat is None:
        if init_method == "pca":
            Y = pca_torch(X, no_dims)
        elif init_method == "rand":
            Y = torch.randn(n, no_dims)
    else:
        #@TODO probably need another hyper-parameter here decide how much we take from previous
        Y = torch.randn(n, no_dims)
        new_idx = min(window_size - jump_size + 1, len(X))
        if init_method == "pca":
            # @TODO: which one more reasonable
            #Y[new_idx - 1:] = pca_torch(X[new_idx-1:], no_dims)
            Y = pca_torch(X, no_dims)
        Y[:min(window_size - jump_size + 1, len(X))] = prev_feat[jump_size:]


    (n, d) = X.shape
    dY = torch.zeros(n, no_dims)
    iY = torch.zeros(n, no_dims)
    gains = torch.ones(n, no_dims)

    # Compute P-values
    #P = x2p_torch(X, tol, perplexity)
    P = torch.randn(n, n)
    P = P + P.t()
    P = P / torch.sum(P)
    P = P * exag_factor    # early exaggeration
    P = torch.max(P, torch.tensor([1e-21]))

    # Run iterations
    for iter in tqdm(range(max_iter), total=max_iter, position=0, leave=True):

        # Compute pairwise affinities
        sum_Y = torch.sum(Y*Y, 1)
        num = -2. * torch.mm(Y, Y.t())
        num = 1. / (1. + torch.add(torch.add(num, sum_Y).t(), sum_Y))
        num[range(n), range(n)] = 0.
        Q = num / torch.sum(num)
        Q = torch.max(Q, torch.tensor([1e-12]))

        # Compute gradient
        PQ = P - Q

        for i in range(n):
            dY[i, :] = torch.sum((PQ[:, i] * num[:, i]).repeat(no_dims, 1).t() * (Y[i, :] - Y), 0)
        # fully-vectorized, large speed up for smaller n
        '''
        temp_a = Y.unsqueeze(1).repeat(1, n, 1).view(-1, no_dims)
        temp_b = Y.tile((n, 1))
        temp_c = ((PQ.T[:n] * num).reshape(-1, 1).repeat(1, no_dims).view(n * n,no_dims) * (temp_a - temp_b))
        dY = torch.sum(temp_c.view(n, n, no_dims), 1)
        '''

        # Perform the update
        if iter < initial_iter:
            momentum = initial_momentum
        else:
            momentum = final_momentum

        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)).double() + (gains * 0.8) * ((dY > 0.) == (iY > 0.)).double()
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - torch.mean(Y, 0)

        # Compute current value of cost function
        if (iter + 1) % 100 == 0:
            C = torch.sum(P * torch.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))

        # Stop lying about P-values
        if iter == early_exag:
            P = P / exag_factor
        torch.cuda.empty_cache()

    # Return solution
    return Y

def get_window_idx(window_size, jump_size, cur_window_idx):

    return start, end

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="wiki40b_30k_reps.dat", help="file name of feature stored")
    parser.add_argument("--cuda", type=int, default=1, help="if use cuda accelarate")

    parser.add_argument("--input_dim", type=int, default=128, help="input dimension")
    parser.add_argument("--output_dim", type=int, default=64, help="output dimension")
    parser.add_argument("--initial_momentum", type=float, default=0.5, help="initial momentum")
    parser.add_argument("--initial_iter", type=int, default=20, help="number of beginning interations to apply intial_momentum ")
    parser.add_argument("--early_exag", type=int, default=100, help="number of beginning interations to apply P value exaggeration ")
    parser.add_argument("--final_momentum", type=float, default=0.8, help="final momentum")
    parser.add_argument("--exag_factor", type=float, default=4.0, help="exag factor for P value for first early_exag iterations")


    parser.add_argument("--output_path", type=str, default="wiki30k_reduced.dat", help="file name of output path")
    parser.add_argument("--init_method", type=str, default="pca", help="pca / random initialization of reduced embedding")

    parser.add_argument("--window_size", type=int, default=5000, help="window size for each tsne computation")
    parser.add_argument("--jump_size", type=int, default=1000, help="(window - overlap size) for each tsne computation")
    parser.add_argument("--perplexity", type=float, default=30.0, help="perplexity")


    parser.add_argument("--n", type=int, default=29727, help="number of data points")
    opt = parser.parse_args()
    print("get choice from args", opt)
    data = opt.data
    jump_size = opt.jump_size
    N = opt.n
    window_size = opt.window_size



    X = np.memmap(opt.data, dtype='float32', mode='r', shape=(opt.n, opt.input_dim))
    # @TODO, no sure if this tensor is too large? Probably we need to use CPU first
    X = torch.Tensor(X)


    # for test-use only
    #X = torch.randn((1800, 128))
    print(X.shape)

    Y = torch.Tensor()
    b_s = (N - window_size) // jump_size + 2
    print(f"Batches to run: {b_s}")
    cur_Y = None

    with torch.no_grad():
        for batch_idx in tqdm(range(b_s), total=b_s, position=0, leave=True):
            start = batch_idx * jump_size
            end = min(start + window_size, opt.n)
            cur_Y = tsne(X[start:end], window_size, jump_size, prev_feat=cur_Y)
            Y = torch.cat((Y, cur_Y))
            if end == opt.n: break

    path = f"ws_{window_size}_js_{jump_size}_init_{opt.init_method}"
    # save output
    torch.save(Y, f"{path}.pt")
    #np.memmap(Y, dtype='float32', mode='r', shape=(opt.n, opt.output))


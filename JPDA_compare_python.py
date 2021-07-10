# -*- coding: utf-8 -*-
# @Time    : 2021/7/10 16:01
# @Author  : Wen Zhang
# @File    : JPDA_compare_python.py
# Reference: https://github.com/jindongwang/transferlearning/blob/master/code/traditional/JDA

import numpy as np
import scipy.io
import scipy.linalg
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import linear_kernel, rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import OneHotEncoder


def kernel(ker, X1, X2, gamma):
    K = None
    if not ker or ker == 'primal':
        K = X1
    elif ker == 'linear':
        if X2:
            K = linear_kernel(np.asarray(X1).T, np.asarray(X2).T)
        else:
            K = linear_kernel(np.asarray(X1).T)
    elif ker == 'rbf':
        if X2:
            K = rbf_kernel(np.asarray(X1).T, np.asarray(X2).T, gamma)
        else:
            K = rbf_kernel(np.asarray(X1).T, None, gamma)
    return K


def get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, mu, type='djp-mmd'):
    M = 0
    if type == 'jmmd':
        N = 0
        n = ns + nt
        e = np.vstack((1 / ns * np.ones((ns, 1)), -1 / nt * np.ones((nt, 1))))
        M0 = e * e.T * C
        if Y_tar_pseudo is not None and len(Y_tar_pseudo) == nt:
            for c in range(1, C + 1):
                e = np.zeros((n, 1))
                tt = Ys == c
                e[np.where(tt == True)] = 1 / len(Ys[np.where(Ys == c)])
                yy = Y_tar_pseudo == c
                ind = np.where(yy == True)
                inds = [item + ns for item in ind]
                e[tuple(inds)] = -1 / len(Y_tar_pseudo[np.where(Y_tar_pseudo == c)])
                e[np.isinf(e)] = 0
                N = N + np.dot(e, e.T)
        M = M0 + N
        M = M / np.linalg.norm(M, 'fro')

    if type == 'jp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        M = Rmin / np.linalg.norm(Rmin, 'fro')

    if type == 'djp-mmd':
        ohe = OneHotEncoder()
        ohe.fit(np.unique(Ys).reshape(-1, 1))
        Ys_ohe = ohe.transform(Ys.reshape(-1, 1)).toarray().astype(np.int8)

        # For transferability
        Ns = 1 / ns * Ys_ohe
        Nt = np.zeros([nt, C])
        if Y_tar_pseudo is not None:
            Yt_ohe = ohe.transform(Y_tar_pseudo.reshape(-1, 1)).toarray().astype(np.int8)
            Nt = 1 / nt * Yt_ohe
        Rmin = np.r_[np.c_[np.dot(Ns, Ns.T), np.dot(-Ns, Nt.T)], np.c_[np.dot(-Nt, Ns.T), np.dot(Nt, Nt.T)]]
        Rmin = Rmin / np.linalg.norm(Rmin, 'fro')

        # For discriminability
        Ms = np.zeros([ns, (C - 1) * C])
        Mt = np.zeros([nt, (C - 1) * C])
        for i in range(C):
            idx = np.arange((C - 1) * i, (C - 1) * (i + 1))
            Ms[:, idx] = np.tile(Ns[:, i], (C - 1, 1)).T
            tmp = np.arange(C)
            Mt[:, idx] = Nt[:, tmp[tmp != i]]
        Rmax = np.r_[np.c_[np.dot(Ms, Ms.T), np.dot(-Ms, Mt.T)], np.c_[np.dot(-Mt, Ms.T), np.dot(Mt, Mt.T)]]
        Rmax = Rmax / np.linalg.norm(Rmax, 'fro')
        M = Rmin - mu * Rmax

    return M


class DA_statistics:
    def __init__(self, kernel_type='primal', mmd_type='djp-mmd', dim=30, lamb=1, gamma=1, mu=0.1, T=5):
        '''
        Init func
        :param kernel_type: kernel, values: 'primal' | 'linear' | 'rbf'
        :param dim: dimension after transfer
        :param lamb: lambda value in equation
        :param gamma: kernel bandwidth for rbf kernel
        :param T: iteration number
        '''
        self.kernel_type = kernel_type
        self.mmd_type = mmd_type
        self.dim = dim
        self.lamb = lamb
        self.gamma = gamma
        self.mu = mu
        self.T = T

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Transform and Predict using 1NN as JDA paper did
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: acc, y_pred, list_acc
        '''
        X = np.hstack((Xs.T, Xt.T))
        X = np.dot(X, np.diag(1. / np.linalg.norm(X, axis=0)))
        m, n = X.shape  # 800, 2081
        ns, nt = len(Xs), len(Xt)

        C = len(np.unique(Ys))
        H = np.eye(n) - 1 / n * np.ones((n, n))

        Y_tar_pseudo = None
        list_acc = []
        for itr in range(self.T):
            M = get_matrix_M(Ys, Y_tar_pseudo, ns, nt, C, self.mu, type=self.mmd_type)

            K = kernel(self.kernel_type, X, None, gamma=self.gamma)
            n_eye = m if self.kernel_type == 'primal' else n
            a, b = np.linalg.multi_dot([K, M, K.T]) + self.lamb * np.eye(n_eye), np.linalg.multi_dot([K, H, K.T])
            w, V = scipy.linalg.eig(a, b)
            ind = np.argsort(w)
            A = V[:, ind[:self.dim]]
            Z = np.dot(A.T, K)
            Z /= np.linalg.norm(Z, axis=0)
            Xs_new, Xt_new = Z[:, :ns].T, Z[:, ns:].T

            clf = KNeighborsClassifier(n_neighbors=1)
            clf.fit(Xs_new, Ys.ravel())
            Y_tar_pseudo = clf.predict(Xt_new)
            acc = accuracy_score(Yt, Y_tar_pseudo)
            list_acc.append(acc)
            print('iteration [{}/{}]: acc: {:.4f}'.format(itr + 1, self.T, acc))
        return list_acc[-1], Y_tar_pseudo, list_acc


if __name__ == '__main__':
    domains = ['caltech_SURF_L10.mat', 'amazon_SURF_L10.mat', 'webcam_SURF_L10.mat', 'dslr_SURF_L10.mat']
    name_list = [name[0].upper() for name in domains]
    mmd_list = ['jmmd', 'jp-mmd', 'djp-mmd']

    num_domain = len(domains)
    acc_all = np.zeros([len(name_list) * (len(name_list) - 1), len(mmd_list)])
    itr_idx = 0
    for s in range(num_domain):  # source
        for t in range(num_domain):  # target
            if s != t:
                print('%s: %s --> %s' % (itr_idx, name_list[s], name_list[t]))
                src, tar = 'data/Office/' + domains[s], 'data/Office/' + domains[t]
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['fts'], src_domain['labels'], tar_domain['fts'], tar_domain['labels']

                # can only be added in offline learning, follow JDA original code
                Xs = preprocessing.scale(Xs)
                Xt = preprocessing.scale(Xt)

                # linear kernel for office-caltech as the original JDA paper
                ker_type = 'linear'

                # # I: joint MMD
                mmd_type = mmd_list[0]
                traditional_tl = DA_statistics(kernel_type=ker_type, mmd_type=mmd_type, dim=100, lamb=1, gamma=1)
                acc_all[itr_idx, 0], _, _ = traditional_tl.fit_predict(Xs, Ys, Xt, Yt)
                print('type: {} -- acc: {:.4f}\n'.format(mmd_type, acc_all[itr_idx, 0]))

                # # II: joint probability MMD
                mmd_type = mmd_list[1]
                traditional_tl = DA_statistics(kernel_type=ker_type, mmd_type=mmd_type, dim=100, lamb=1, gamma=1)
                acc_all[itr_idx, 1], _, _ = traditional_tl.fit_predict(Xs, Ys, Xt, Yt)
                print('type: {} -- acc: {:.4f}\n'.format(mmd_type, acc_all[itr_idx, 1]))

                # # III: discriminative joint probability MMD
                mmd_type = mmd_list[2]
                traditional_tl = DA_statistics(kernel_type=ker_type, mmd_type=mmd_type, dim=100, lamb=1, gamma=1)
                acc_all[itr_idx, 2], _, _ = traditional_tl.fit_predict(Xs, Ys, Xt, Yt)
                print('type: {} -- acc: {:.4f}\n'.format(mmd_type, acc_all[itr_idx, 2]))
                itr_idx += 1
    
    print('mean acc...')
    print(np.round(np.mean(acc_all, axis=0), 4))
    print('\n', mmd_list)
    print(np.round(acc_all, 4))

# mean acc...
# [0.4549 0.4634 0.4798]
#
# ['jmmd', 'jp-mmd', 'djp-mmd']
# [[0.4551 0.4499 0.4854]
#  [0.4102 0.4203 0.4441]
# [0.3822 0.3822 0.4268]
# [0.4025 0.39   0.3954]
# [0.4034 0.3966 0.4441]
# [0.4204 0.4459 0.4459]
# [0.3161 0.3197 0.3179]
# [0.3173 0.2954 0.3163]
# [0.8917 0.9108 0.8981]
# [0.2965 0.3375 0.3571]
# [0.3257 0.3351 0.3486]
# [0.8373 0.878  0.878]]

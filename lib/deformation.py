import networkx as nx
import copy
import scipy
import scipy.sparse
from scipy.sparse.linalg import lsqr
from scipy.sparse import csr_matrix
import numpy as np

from lib.graph_helper import GraphHelper as GH


class LaplacianDeformer:

    def __init__(self, G:nx.Graph, anchor_idx, weights=None):
        # G may be a subgraph with disordered indices
        self.G = G
        self.g2l = {node: i for i, node in enumerate(G.nodes)}  # global to local
        self.l2g = list(G.nodes)  # local to global
        self.anchor_idx = np.array(anchor_idx)

        self.weights = np.ones((len(anchor_idx), )) if weights is None else weights

        for node in G.nodes:
            G.nodes[node]['anchor'] = False

        for g in anchor_idx:
            G.nodes[g]['anchor'] = True

    def deform(self, anchor_idx, anchor_co:np.ndarray):
        """
        :param anchor_idx: global anchor idx
        :param anchor_co:
        :return:
        """
        L = GH.rw_laplacian_matrix(self.G) # (n, n)
        n = L.shape[0]
        m = len(self.anchor_idx)

        # construct lambda matrix
        rows = np.arange(len(self.anchor_idx))
        cols = np.array([self.g2l[g] for g in self.anchor_idx])
        lambda_ = csr_matrix((self.weights, (rows, cols)), shape=(m, n))        # (m, n)

        L = scipy.sparse.vstack((L, lambda_))
        V = np.array([self.G.nodes[i]['co'] for i in self.G.nodes])
        delta = L.dot(V)

        # add constraints
        for i, g in enumerate(anchor_idx):
            delta[n+i, :] = anchor_co[i] * self.weights[i]

        delta = -delta

        # solve
        A = scipy.sparse.block_diag((L, L, L))
        b = np.hstack((delta[:, 0], delta[:, 1], delta[:, 2]))
        ans = lsqr(A, b)        # (n, 3)

        return ans[0].reshape(-1, 3)

    def deform_iter(self, anchor_idx, anchor_co, iter=1):
        G = self.G.copy()

        # calc initial delta
        for g in G.nodes:
            G.nodes[g]['delta'] = G.nodes[g]['co'] - np.array([G.nodes[g_ne]['co'] for g_ne in G.neighbors(g)]).mean(0)

        for g, co in zip(anchor_idx, anchor_co):
            G.nodes[g]['co'] = co

        # iteration
        for _ in range(iter):
            for i in G.nodes:
                G.nodes[i]['visited'] = False
            stack_g = list(anchor_idx)
            for g in stack_g:
                G.nodes[g]['visited'] = True

            while len(stack_g)>0:
                g = stack_g[0]
                stack_g.remove(g)
                for g_ne in G.neighbors(g):
                    if not G.nodes[g_ne]['visited']:
                        stack_g.append(g_ne)
                        G.nodes[g_ne]['visited'] = True

                if not G.nodes[g]['anchor']:
                    _delta = G.nodes[g]['co'] - np.array([G.nodes[g_ne]['co'] for g_ne in G.neighbors(g)]).mean(0)
                    if np.all(np.abs(G.nodes[g]['delta'] - _delta) < 1e-4):
                        break
                    G.nodes[g]['co'] += G.nodes[g]['delta'] - _delta

        return np.array([G.nodes[i]['co'] for i in G.nodes])

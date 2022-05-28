import networkx as nx
import scipy
import scipy.sparse
from typing import List

class GraphHelper:

    @staticmethod
    def build_vert_graph(verts, edges):
        G = nx.Graph()
        for i, v in enumerate(verts):
            G.add_node(i, co=v)

        for e in edges:
            G.add_edge(*e)
        return G

    @staticmethod
    def rw_laplacian_matrix(G:nx.Graph):
        # randow walk laplacian matrix
        # lap = nx.laplacian_matrix(G)
        A = nx.to_scipy_sparse_matrix(G, weight='weight', format="csr")
        n, m = A.shape
        diags = A.sum(axis=1)
        inv_D = scipy.sparse.spdiags(1. / diags.flatten(), [0], m, n, format="csr")

        return scipy.sparse.eye(m, n) - inv_D.dot(A)

    @staticmethod
    def get_boundary_by_ctrl_points(G:nx.Graph, ctrl_points_idx)->List[int]:
        boundary = []
        for i in range(len(ctrl_points_idx)):
            p1 = ctrl_points_idx[i]
            p2 = ctrl_points_idx[(i + 1) % len(ctrl_points_idx)]
            path = nx.shortest_path(G, p1, p2)
            boundary += path[0:-1]
        return boundary

    @staticmethod
    def get_subgraph(G:nx.Graph, boundary, manip_handle)->nx.Graph:
        g_part = G.copy()
        g_part.remove_nodes_from(boundary)
        nodes = list(nx.node_connected_component(g_part, manip_handle))
        return G.subgraph(boundary + nodes)


import torch
from torch import Tensor
from scipy.spatial import Delaunay
from scipy.spatial.qhull import QhullError

import itertools
import numpy as np


def build_graphs(P_np: np.ndarray, In_np: np.ndarray, n: int, n_pad: int=None, edge_pad: int=None, stg: str='fc'):
    """
    Build graph matrix G,H from point set P. This function supports only cpu operations in numpy.
    G, H is constructed from adjacency matrix A: A = G * H^T
    :param P_np: point set containing point coordinates
    :param n: number of exact points in the point set
    :param n_pad: padded node length
    :param edge_pad: padded edge length
    :param stg: strategy to build graphs.
                'tri', apply Delaunay triangulation or not.
                'near', fully-connected manner, but edges which are longer than max(w, h) is removed
                'fc'(default), a fully-connected graph is constructed
    :param device: device. If not specified, it will be the same as the input
    :return: G, H, edge_num
    """

    assert stg in ('fc', 'tri', 'near', 'fps', 'nnweight', 'inlier'), 'No strategy named {} found.'.format(stg)

    if stg == 'tri':
        A = delaunay_triangulate(P_np[0:n, :])
    elif stg == 'near':
        A = fully_connect(P_np[0:n, :], thre=0.2)
    elif stg == 'fps':
        A = fpsknn_connect(P_np[0:n, :], 50,  thre=0.2)
    elif stg == 'nnweight':
        A = nn_weight_connect(P_np[0:n, :], thre=0.1)
    elif stg == 'inlier':
        A = inlier_connect(P_np[0:n, :], In_np[0:n, :])
    else:
        A = fully_connect(P_np[0:n, :])
    edge_num = int(np.sum(A, axis=(0, 1)))
    assert n > 0 and edge_num > 0, 'Error in n = {} and edge_num = {}'.format(n, edge_num)

    if n_pad is None:
        n_pad = n
    if edge_pad is None:
        edge_pad = edge_num
    assert n_pad >= n
    assert edge_pad >= edge_num

    return A, edge_num


def delaunay_triangulate(P: np.ndarray):
    """
    Perform delaunay triangulation on point set P.
    :param P: point set
    :return: adjacency matrix A
    """
    n = P.shape[0]
    if n < 3:
        A = fully_connect(P)
    else:
        try:
            d = Delaunay(P)
            #assert d.coplanar.size == 0, 'Delaunay triangulation omits points.'
            A = np.zeros((n, n))
            for simplex in d.simplices:
                for pair in itertools.permutations(simplex, 2):
                    A[pair] = 1
        except QhullError as err:
            print('Delaunay triangulation error detected. Return fully-connected graph.')
            print('Traceback:')
            print(err)
            A = fully_connect(P)
    return A


def fully_connect(P: np.ndarray, thre=None):
    """
    Fully connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        xyz = P[:, :3]
        dist = -2 * xyz @ xyz.T
        dist += np.sum(xyz ** 2, axis=-1)[:, None]
        dist += np.sum(xyz ** 2, axis=-1)[None, :]
        PP_dist_flag = dist > (thre**2)
        # P_rep = np.expand_dims(P[:, :3], axis=1).repeat(n, axis=1)
        # PP_dist_flag = np.sqrt(np.sum(np.square(P_rep - P[None,:, :3]), axis=2)) > thre
        A[PP_dist_flag] = 0
        # for i in range(n):
        #     for j in range(i):
        #         if np.linalg.norm(P[i] - P[j]) > thre:
        #             A[i, j] = 0
        #             A[j, i] = 0
    return A


def nn_weight_connect(P: np.ndarray, thre=None):
    """
    nn_weight_connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    if thre is not None:
        xyz = P[:, :3]
        dist = -2 * xyz @ xyz.T
        dist += np.sum(xyz ** 2, axis=-1)[:, None]
        dist += np.sum(xyz ** 2, axis=-1)[None, :]
        PP_dist_flag = dist > (thre**2)
        A_weight = 1/(((dist[PP_dist_flag]**0.5)*10))
        A[PP_dist_flag] = A_weight
    return A


def inlier_connect(P: np.ndarray, inlier: np.ndarray):
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    A[inlier.squeeze() == 0] = 0
    A[:, inlier.squeeze() == 0] = 0
    return A


def fpsknn_connect(P: np.ndarray, K=20,  thre=None):
    """
    Fully connect a graph.
    :param P: point set
    :param thre: edges that are longer than this threshold will be removed
    :return: adjacency matrix A
    """
    n = P.shape[0]
    A = np.ones((n, n)) - np.eye(n)
    indexs, _ = farthest_point_sampling(P, K)
    if thre is not None:
        for i in range(n):
            for j in range(i):
                if np.linalg.norm(P[i] - P[j]) > thre:
                    A[i, j] = 0
                    A[j, i] = 0
                if j in indexs:
                    A[i, j] = 1
                    A[j, i] = 1
    return A


def l2_norm(x, y):
    """Calculate l2 norm (distance) of `x` and `y`.
    Args:
        x (numpy.ndarray or cupy): (batch_size, num_point, coord_dim)
        y (numpy.ndarray): (batch_size, num_point, coord_dim)
    Returns (numpy.ndarray): (batch_size, num_point,)
    """
    return ((x - y) ** 2).sum(axis=2)


def farthest_point_sampling(pts, k, initial_idx=None, metrics=l2_norm,
                            skip_initial=False, indices_dtype=np.int32,
                            distances_dtype=np.float32):
    """Batch operation of farthest point sampling
    Code referenced from below link by @Graipher
    https://codereview.stackexchange.com/questions/179561/farthest-point-algorithm-in-python
    Args:
        pts (numpy.ndarray or cupy.ndarray): 2-dim array (num_point, coord_dim)
            or 3-dim array (batch_size, num_point, coord_dim)
            When input is 2-dim array, it is treated as 3-dim array with
            `batch_size=1`.
        k (int): number of points to sample
        initial_idx (int): initial index to start farthest point sampling.
            `None` indicates to sample from random index,
            in this case the returned value is not deterministic.
        metrics (callable): metrics function, indicates how to calc distance.
        skip_initial (bool): If True, initial point is skipped to store as
            farthest point. It stabilizes the function output.
        xp (numpy or cupy):
        indices_dtype (): dtype of output `indices`
        distances_dtype (): dtype of output `distances`
    Returns (tuple): `indices` and `distances`.
        indices (numpy.ndarray or cupy.ndarray): 2-dim array (batch_size, k, )
            indices of sampled farthest points.
            `pts[indices[i, j]]` represents `i-th` batch element of `j-th`
            farthest point.
        distances (numpy.ndarray or cupy.ndarray): 3-dim array
            (batch_size, k, num_point)
    """
    if pts.ndim == 2:
        # insert batch_size axis
        pts = pts[None, ...]
    assert pts.ndim == 3
    batch_size, num_point, coord_dim = pts.shape
    indices = np.zeros((batch_size, k, ), dtype=indices_dtype)

    # distances[bs, i, j] is distance between i-th farthest point `pts[bs, i]`
    # and j-th input point `pts[bs, j]`.
    distances = np.zeros((batch_size, k, num_point), dtype=distances_dtype)
    if initial_idx is None:
        indices[:, 0] = np.random.randint(len(pts))
    else:
        indices[:, 0] = initial_idx

    batch_indices = np.arange(batch_size)
    farthest_point = pts[batch_indices, indices[:, 0]]
    # minimum distances to the sampled farthest point
    try:
        min_distances = metrics(farthest_point[:, None, :], pts)
    except Exception as e:
        import IPython; IPython.embed()

    if skip_initial:
        # Override 0-th `indices` by the farthest point of `initial_idx`
        indices[:, 0] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, 0]]
        min_distances = metrics(farthest_point[:, None, :], pts)

    distances[:, 0, :] = min_distances
    for i in range(1, k):
        indices[:, i] = np.argmax(min_distances, axis=1)
        farthest_point = pts[batch_indices, indices[:, i]]
        dist = metrics(farthest_point[:, None, :], pts)
        distances[:, i, :] = dist
        min_distances = np.minimum(min_distances, dist)
    return indices, distances

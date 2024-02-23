from collections import Counter
from pathlib import Path
import pickle

from multiprocessing import Pool
import networkx as nx
import numpy as np
import pandas as pd
import potpourri3d as pp3d
from pygeodesic import geodesic
from scipy.special import softmax
import trimesh
from trimesh.exchange.ply import load_ply as tri_load_ply



###############################
### Load data from ply files

def map_coords(xyz1, xyz2):
    key = {i:j for i, j in zip(*[np.argsort(x.sum(axis=1)) for x in [xyz1, xyz2]])}
    return np.array([key[i] for i in range(len(xyz1))])


def load_ply(path):
    return trimesh.load(open(path, 'r'), 'ply')


def load_ply_feat(path):
    mesh = load_ply(path)
    N = len(mesh.vertices)

    ply  = tri_load_ply(open(path, 'r'))
    xyz = np.array([ply['metadata']['_ply_raw']['vertex']['data'][x] for x in 'xyz']).T.reshape(N,3)
    idx_map = map_coords(mesh.vertices, xyz)

    feat = {k:v.reshape(N)[idx_map] for k, v in ply['metadata']['_ply_raw']['vertex']['data'].items()}
    return mesh, feat


###############################
### Calculate geodesic distance

def generate_graph_from_mesh(mesh, idx=''):
    if isinstance(idx, str):
        edges = mesh.edges_unique
        length = mesh.edges_unique_length
    else:
        edges = mesh.edges_unique[idx]
        length = mesh.edges_unique_length[idx]
    N = len(mesh.vertices)

    g = nx.Graph()
    for edge, L in zip(edges, length):
        g.add_edge(*edge, length=L)

    return g


def unpack_path_lengths(paths, N):
    arr = np.zeros((N,N), dtype=float) - 1
    for i, v in paths:
        for j, l in v.items():
            arr[i,j] = l
    return arr


def geodesic_distance_matrix(mesh, idx='', weight='length'):
    g = generate_graph_from_mesh(mesh, idx=idx)
    if weight:
        path_length = nx.all_pairs_dijkstra_path_length(g, weight=weight)
    else:
        path_length = nx.all_pairs_dijkstra_path_length(g)
    if isinstance(idx, str):
        return unpack_path_lengths(path_length, len(mesh.vertices))
    else:
        return unpack_path_lengths(path_length, len(idx))



###############################
### Run analyses


def get_residue_distmat(path_ply):
    mesh, feat = load_ply_feat(path_ply)
#   dist = pp3d.compute_distance_multisource(mesh.vertices, mesh.faces, range(len(mesh.vertices)))
    dist = geodesic_distance_matrix(mesh)

#   N = len(mesh.vertices)
#   idx = np.arange(N)
#   i, j = [x.ravel() for x in np.meshgrid(idx, idx)]
#   print(mesh.vertices.shape, mesh.faces.shape)
#   geoalg = geodesic.PyGeodesicAlgorithmExact(mesh.vertices, mesh.faces)
#   dist = geoalg.geodesicDistances(i,j)[0].reshape(N, N)

    residx = feat['residx'].astype(int)
    # Re-order the indices to start at zero
    residx -= 1

    i_list = np.unique(residx)
    N = residx.max() + 1
    res_dist = np.zeros((N, N), float)

    for i in range(N - 1):
        i0 = residx[i]
        idx1 = residx == i0
        for j in range(i + 1, N):
            j0 = residx[j]
            idx2 = residx == j0
            d = np.mean(dist[idx1][:,idx2])
            res_dist[i0,j0] = d
            res_dist[j0,i0] = d
    return res_dist


def get_reference_distmat():
    path_ply = "2i9w_A_0000.ply"
    dist = get_residue_distmat(path_ply)
    np.save("2i9w_A_resdist.ply", dist)


def get_gini(prob):
    prob = np.array(sorted(prob))
    cumprob = np.cumsum(prob / prob.sum())
    diag = np.arange(1, prob.size + 1) / prob.size
    return np.sum((diag - cumprob)) / np.sum(diag)


def get_gini_list(seq):
    return get_gini(list(Counter(seq).values()))


def evaluate_vertex_p2p_map(dist, p12, residx1, residx2):
    res_dist = dist[residx2, residx1[p12]]
    mean_dist = np.mean(res_dist)
    frac_zero = np.sum(res_dist == 0) / res_dist.size
    gini = get_gini_list(p12)
    print(f"Mean distance: {mean_dist:5.2f}")
    print(f"Frac zero: {frac_zero:5.2f}")
    print(f"Gini coefficient: {gini:5.2f}")
    return mean_dist, frac_zero, gini


def get_residx(path_ply):
    mesh, feat = load_ply_feat(path_ply)
    residx = feat['residx'].astype(int)
    # Re-order the indices to start at zero
    residx -= 1
    return residx










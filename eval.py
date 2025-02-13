import argparse
import yaml
from pathlib import Path
import os
import torch
from scape_dataset import ScapeDataset, shape_to_device
# from dt4d_dataset import ScapeDataset, shape_to_device
from model import DQFMNet
#
import numpy as np
from pyFM.spectral.nn_utils import knn_query
import scipy.io as sio
from scipy.spatial.distance import cdist
from scipy.special import softmax
from tqdm import tqdm
from utils import read_geodist, augment_batch, augment_batch_sym
from Tools.utils import fMap2pMap, zo_fmap
from diffusion_net.utils import toNP
import torch.nn.functional as F

import evaluate_map as EM

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    bs, m, n = x.size(0), x.size(1), y.size(1)
    xx = torch.pow(x, 2).sum(2, keepdim=True).expand(bs, m, n)
    yy = torch.pow(y, 2).sum(2, keepdim=True).expand(bs, n, m).transpose(1, 2)
    dist = xx + yy - 2 * torch.bmm(x, y.transpose(1, 2))
    # dist = dist.clamp(min=1e-12).sqrt() 
    return dist


# It is equal to Tij = knnsearch(j, i) in Matlab
def knnsearch(x, y, alpha):
    distance = cdist(x, y)
    output = softmax(-alpha*distance, axis=-1)
    return output


def convert_C(C12, Phi1, Phi2, alpha):
    Phi1, Phi2 = Phi1[:, :50], Phi2[:, :50]
    T21 = knnsearch(Phi2 @ C12, Phi1, alpha)
    C12_new = np.linalg.pinv(Phi2) @ (T21 @ Phi1)
    return C12_new


def load_protein_geodist(cfg, shape_name):
    prot_name = '_'.join(shape_name.split('_')[:2])
    path_geo = os.path.join(cfg['dataset']['root_dataset'], "geodesic", f"{prot_name}.npy")
    if os.path.exists(path_geo):
        return np.load(path_geo)

    path_ply = os.path.join(cfg['dataset']['root_dataset'], cfg['dataset']['name'], "shapes_train", f"{prot_name}_0000.ply")
    dist = EM.get_residue_distmat(path_ply)
    np.save(path_geo, dist)
    return dist


def eval_geodist(cfg):

    train_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=True)

    shape_names = train_dataset.used_shapes
    prot_uniq = np.unique(['_'.join(s[:2]) for s in shape_names])

    for i, (j, k) in tqdm(enumerate(train_dataset.combinations)):
        geodist = load_protein_geodist(cfg, shape1)

    n_s = G_s.shape[0]
    # print(SQ_s[0])
    if 'vts' in shape1:
        phi_t = shape1['vts']
        phi_s = shape2['vts']
    elif 'gt' in shape1:
        phi_t = np.arange(shape1['xyz'].shape[0])
        phi_s = shape1['gt']
    else:
        raise NotImplementedError("cannot find ground-truth correspondence for eval")

    # find pairs of points for geodesic error
    pmap = T
    ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims=[n_s, n_s])

    errC = np.take(G_s, ind21) / SQ_s
    print('{}-->{}: {:.4f}'.format(shape1['name'], shape2['name'], np.mean(errC)))
    return errC

def eval_net(args, model_path):
    if torch.cuda.is_available() and cfg["misc"]["cuda"]:
        device = torch.device(f'cuda:{cfg["misc"]["device"]}')
    else:
        device = torch.device("cpu")

    # important paths
    base_path = os.path.dirname(__file__)
    op_cache_dir = os.path.join(base_path, cfg["dataset"]["cache_dir"])
    dataset_path = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["root_test"])

    # decide on the use of WKS descriptors
    with_wks = None if cfg["fmap"]["C_in"] <= 3 else cfg["fmap"]["C_in"]

    # create dataset
    train_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
                                    k_eig=cfg["fmap"]["k_eig"],
                                    n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
                                    with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
                                    use_cache=True, op_cache_dir=op_cache_dir, train=True)
#     if cfg["dataset"]["type"] == "vts":
#         test_dataset = ScapeDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
#                                     k_eig=cfg["fmap"]["k_eig"],
#                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
#                                     with_wks=with_wks, with_sym=cfg["dataset"]["with_sym"],
#                                     use_cache=True, op_cache_dir=op_cache_dir, train=False)

#     elif cfg["dataset"]["type"] == "gt":
#         test_dataset = ShrecDataset(dataset_path, name=cfg["dataset"]["name"] + "-" + cfg["dataset"]["subset"],
#                                     k_eig=cfg["fmap"]["k_eig"],
#                                     n_fmap=cfg["fmap"]["n_fmap"], n_cfmap=cfg["fmap"]["n_cfmap"],
#                                     with_wks=with_wks,
#                                     use_cache=True, op_cache_dir=op_cache_dir, train=False)

    # else:
    #     raise NotImplementedError("dataset not implemented!")

    prot_geodist = {}

    # test loader
    test_loader = torch.utils.data.DataLoader(train_dataset, batch_size=None, shuffle=False)

    # define model
    dqfm_net = DQFMNet(cfg).to(device)
    model_path = os.path.join(cfg["dataset"]["root_dataset"], f'trained_{cfg["dataset"]["name"]}', model_path)
#   model_path = "data/trained_prot3/ep_val_best.pth"
    dqfm_net.load_state_dict(torch.load(model_path, map_location=device))
    dqfm_net.eval()

    result = []

    for i, data in tqdm(enumerate(test_loader)):
        data = shape_to_device(data, device)

        # data augmentation (if using wks descriptors augment with sym)
        # if with_wks is None:
        #     data = augment_batch(data, rot_x=180, rot_y=180, rot_z=180,
        #                          std=0.01, noise_clip=0.05,
        #                          scale_min=0.9, scale_max=1.1)
        # elif "with_sym" in cfg["dataset"] and cfg["dataset"]["with_sym"]:
        #     data = augment_batch_sym(data, rand=False)

        # prepare iteration data

        # do iteration
        C_pred = dqfm_net(data)[0]
        Phi1, Phi2 = data["shape1"]['evecs'], data["shape2"]['evecs']

        # check rank
        # print(feat1.shape)
        # feat1, feat2 = feat1.cpu().numpy(), feat2.cpu().numpy()
        # rank1, rank2 = torch.linalg.matrix_rank(feat1.squeeze()), torch.linalg.matrix_rank(feat2.squeeze())
        #print([rank1, rank2])

        # save maps
        name1, name2 = data["shape1"]["name"], data["shape2"]["name"]


        save_path = os.path.join(cfg["dataset"]["root_dataset"], f'results_{cfg["dataset"]["name"]}')
        save_path_c = save_path + '/C/'
        if not os.path.exists(save_path_c):
            os.makedirs(save_path_c)

        filename_c12 = f'C_{name1}_{name2}.mat'
        c12 = C_pred.detach().cpu().squeeze(0).numpy()
        c12_dic = {'C': c12}
        sio.savemat(os.path.join(save_path_c, filename_c12), c12_dic)

        # compute geodesic error (transpose C12 to get C21, and thus T12)
        shape1, shape2 = data["shape1"], data["shape2"]

        save_path_phi = save_path + '/Phi/'
        if not os.path.exists(save_path_phi):
            os.makedirs(save_path_phi)

        filename_phi1 = f'Phi_{name1}.mat'
        Phi1 = toNP(shape1['evecs'])
        Phi1_dic = {'Phi': Phi1}
        sio.savemat(os.path.join(save_path_phi, filename_phi1), Phi1_dic)


#   shape_names = train_dataset.used_shapes
#   prot_names = {'_'.join(s.split('_')[:2]):s for s in shape_names}
        prot_name = '_'.join(data['shape1']['name'].split('_')[:2])
        if prot_name not in prot_geodist:
            print(prot_name)
            prot_geodist[prot_name] = load_protein_geodist(cfg, prot_name)

        path_ply1 = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["name"], "shapes_train", f"{data['shape1']['name']}.ply")
        path_ply2 = os.path.join(cfg["dataset"]["root_dataset"], cfg["dataset"]["name"], "shapes_train", f"{data['shape2']['name']}.ply")

        ri1 = EM.get_residx(path_ply1)
        ri2 = EM.get_residx(path_ply2)
#       p21 = convert_C(c12, Phi1, Phi2.cpu(), cfg['loss']['max_alpha'])
#       p21 = p21.astype(int)
        p21 = knn_query(Phi1[:,:50], Phi2.cpu()[:,:50] @ c12, k=1)
        result.append(EM.evaluate_vertex_p2p_map(prot_geodist[prot_name], p21, ri1, ri2))

    mean_diff, frac_zero, gini = np.array(result).T
    np.save(os.path.join(save_path, "mean_diff.npy"), mean_diff)
    np.save(os.path.join(save_path, "frac_zero.npy"), frac_zero)
    np.save(os.path.join(save_path, "gini.npy"), gini)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch the eval of DQFM model.")

    parser.add_argument("--config", type=str, default="smal_r", help="Config file name")

    parser.add_argument("--model_path", type=str, default="ep_val_best.pth",
                         help="path to saved model")
    parser.add_argument("--save_path", type=str, default="data/results",
                        help="dir to save C_pred")


    args = parser.parse_args()
    cfg = yaml.safe_load(open(f"./{args.config}.yaml", "r"))
    eval_net(cfg, args.model_path)

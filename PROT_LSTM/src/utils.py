import torch
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence as pack
from torch.autograd import Variable
#from torch import cat, stack, zeros, cuda
#from torch import atan2, stack, sin, cos, Tensor, cross, norm, mm


def stack_pack(var, lengths):
    max_len, n_feats = var[0].size()
    var = [torch.cat((s, torch.zeros(max_len - s.size(0), n_feats)), 0) if s.size(0) != max_len else s for s in var]
    var = torch.stack(var, 0)
    if torch.cuda.is_available():
        var = Variable(var).cuda()
    else:
        var = Variable(var)
    return pack(var, lengths, batch_first=True)


def pad_packed_collate(batch, aa):

    if len(batch) == 1:
        aa, angles, coords = batch[0]
        lengths_aa = [aa.size(0)]
        lengths_coords = [coords.size(0)]
        aa.unsqueeze_(0)
        angles.unsqueeze_(0)
        coords.unsqueeze_(0)
    if len(batch) > 1:
        aa, angles, coords, lengths_aa, lengths_coords = zip(
            *[(a, b, c, a.size(0), c.size(0)) for (a, b, c) in sorted(batch, key=lambda x: x[0].size(0), reverse=True)])
    packed_aa = stack_pack(aa, lengths_aa)
    packed_angles = stack_pack(angles, lengths_aa)
    packed_coords = stack_pack(coords, lengths_coords)
    return packed_aa, packed_angles, packed_coords


def position(A, B, C, bc, R, theta, phi):
    n = torch.cross(B-A, C-B)
    n = n/torch.norm(n)
    D = torch.stack([R*torch.cos(theta), R*torch.sin(theta)*torch.cos(phi), R*torch.sin(theta)*torch.sin(phi)])
    M = torch.stack([(C-B)/bc, torch.cross(n, C-B)/bc, n], dim=1)
    return torch.mm(M,D).squeeze() + C


def reconstruct(ang, init):
    N_Ca = 1.458
    Ca_C = 1.525
    C_N = 1.329
    R = [C_N, N_Ca, Ca_C]
    bond_angles = torch.stack([torch.atan2(ang[:,0], ang[:,1]),
                               torch.atan2(ang[:,2], ang[:,3]),
                               torch.atan2(ang[:,4], ang[:,5])],
                              dim=1).view(-1)
    torsion_angles = torch.stack([torch.atan2(ang[:,6], ang[:,7]),
                                  torch.atan2(ang[:,8], ang[:,9]),
                                  torch.atan2(ang[:,10], ang[:,11])],
                                 dim=1).view(-1)
    if torch.cuda.is_available():
        pos = Variable(torch.Tensor(len(bond_angles),3)).cuda()
    else:
        pos = Variable(torch.Tensor(len(bond_angles),3))
    pos[0] = init[0]
    pos[1] = init[1]
    pos[2] = init[2]
    for ij in range(3, len(bond_angles)):
        pos[ij] = position(pos[ij-3], pos[ij-2], pos[ij-1], R[(ij-1)%3], R[ij%3], (np.pi-bond_angles[ij-1]), torsion_angles[ij-1])
    return bond_angles*180/np.pi, torsion_angles*180/np.pi, pos


def pdist(x):
    x_norm = x.pow(2).sum(1).view(-1, 1)
    y_t = torch.transpose(x, 0, 1)
    y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2 * torch.mm(x, y_t)
    dist = dist - torch.diag(dist.diag())
    dist = torch.clamp(dist, 0.0, np.inf)
    dist = dist.pow(0.5)
    return dist


def rad2deg(rad_angle):
    if rad_angle is None:
        return None

    angle = (rad_angle * 180 / np.pi) % 360

    if angle > 180:
        angle = angle - 360
    elif angle < -180:
        angle = angle + 360

    return angle
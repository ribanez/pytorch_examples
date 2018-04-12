import os
import random
import numpy as np
import pandas as pd
from glob import glob

from torch import FloatTensor
from torch.utils.data import Dataset



class PDBDataset(Dataset):

    def __init__(self, root_dir, aa):
        self.root_dir = root_dir
        self.names = sorted(glob(os.path.join(self.root_dir, '*.csv')))
        self.aa = aa

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):

        idx = random.randint(0, len(self.names))
        df = pd.read_csv(self.names[idx])

        sequence = np.zeros((len(df)//3, 20))
        angles = np.zeros((len(df)//3, 12))

        for i in range(len(df)//3):
            sequence[i, self.aa.index(df.aa[3*i])] = 1
            angles[i, 0] = np.sin(df.bond_angle[3*i]*np.pi/180)
            angles[i, 1] = np.cos(df.bond_angle[3*i]*np.pi/180)
            angles[i, 2] = np.sin(df.bond_angle[3*i+1]*np.pi/180)
            angles[i, 3] = np.cos(df.bond_angle[3*i+1]*np.pi/180)
            angles[i, 4] = np.sin(df.bond_angle[3*i+2]*np.pi/180)
            angles[i, 5] = np.cos(df.bond_angle[3*i+2]*np.pi/180)
            angles[i, 6] = np.sin(df.torsion_angle[3*i]*np.pi/180)
            angles[i, 7] = np.cos(df.torsion_angle[3*i]*np.pi/180)
            angles[i, 8] = np.sin(df.torsion_angle[3*i+1]*np.pi/180)
            angles[i, 9] = np.cos(df.torsion_angle[3*i+1]*np.pi/180)
            angles[i, 10] = np.sin(df.torsion_angle[3*i+2]*np.pi/180)
            angles[i, 11] = np.cos(df.torsion_angle[3*i+2]*np.pi/180)
        coords = np.stack([df.x, df.y, df.z], axis=1)

        return FloatTensor(sequence), FloatTensor(angles), FloatTensor(coords)
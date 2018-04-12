import pandas as pd
from src.utils import rad2deg
from Bio.PDB import PDBParser, calc_angle, calc_dihedral, PPBuilder


def Parser(PDB_file):
    # this script extract all the pdb info and then puts it into a csv file

    df = pd.DataFrame(columns=('CHAIN', 'RES', 'RES_N', 'PHI', 'PSI', 'OMEGA'))
    residue_list = (
    "SEP", "TPO", "ALA", "GLY", "ILE", "LEU", "PRO", "VAL", "PHE", "TRP", "TYR", "ASP", "GLY", "ARG", "HIS", "LYS",
    "SER", "THR", "CYS", "MET", "ASN", "GLN", "GLU", "HOH", "ZN")
    p = PDBParser()
    # Summary_file=open("./Summary.txt","w")
    chains_l = list()
    cont_no_AA = 0
    cont_resid = 0
    x = 1
    ####for PDB_file in PDB_list_file:
    ####PDB_file=PDB_file.replace("\n", "")
    structure = p.get_structure("X", "/mnt/DATA/1/" + PDB_file)  # nombre del archivo PDB a convertir
    model = structure[0]

    for chain in model:  # FOR I
        # print(chain.get_id())
        cont_no_AA = 0
        cont_resid = 0
        for residue in chain:  # FOR II
            cont_resid = cont_resid + 1
            if (residue.get_resname() not in residue_list): cont_no_AA = cont_no_AA + 1
        print(PDB_file, chain.get_id(), cont_resid, cont_no_AA)
        if (cont_no_AA == 0 and cont_resid >= 20): chains_l.append(chain.get_id())

    # in this step we will loop through the valid chains in the models
    for chain in model:
        c_n = str(chain.get_id())
        if (chain.get_id() in chains_l):
            polypeptides = PPBuilder().build_peptides(
                chain)  # obiente TODOS los polipeptidos de la cadena actual
            for poly_index, poly in enumerate(polypeptides):  # poly = residue
                # print (poly.get_phi_psi_list())
                phi_psi = poly.get_phi_psi_list()
                for res_index, residue in enumerate(poly):
                    phi, psi = phi_psi[res_index]
                    if (phi and psi):
                        # print(residue.resname,residue.id[1],degrees(phi),degrees(psi))
                        list_buffer3 = pd.DataFrame(
                            {"CHAIN": c_n, "RES_N": residue.id[1], "RES": residue.resname, "PHI": rad2deg(phi),
                             "PSI": rad2deg(psi)}, index=[0])
                        if (x):
                            result = pd.concat([df, list_buffer3], ignore_index=True, axis=0)
                            x = 0
                        else:
                            result = pd.concat([result, list_buffer3], ignore_index=True, axis=0)

                            result = result[['CHAIN', 'RES', 'RES_N', 'PHI', 'PSI']]
                            result.to_csv(PDB_file[:-4] + ".csv", sep=',', encoding='utf-8', index=False)

    del chains_l[:]  # this clean the chain list per model in FOR I (in line)


# print("FIN")

PDB_list_file = open("/mnt/DATA/1/listA.txt", "r")
for PDB_file in PDB_list_file:
    PDB_file = PDB_file.replace("\n", "")
    Parser(PDB_file)



####### CODIGO DE VERDAD ######
import os
import glob

from Bio.PDB import PDBParser, calc_angle, calc_dihedral
from numpy import isin, NaN
from pandas import DataFrame
from numpy import cross, array, sin, cos, pi, matmul, dot, arccos, stack
from numpy.linalg import norm


def parse(structure):
    num_models = len(structure)
    if num_models > 1:
        print('Number of models ' + str(num_models))
        return DataFrame([])
    num_chains = len(structure[0])
    if num_chains > 1:
        print('Number of chains ' + str(num_chains))
        return DataFrame([])

    df = DataFrame([])
    res = 0
    for residue in structure[0]['A']: #consider only chain A
        if residue.get_id()[0] == ' ': #ignore all hetero atoms
            name = residue.get_resname()
            if not isin(name, aa):
                print('Non recognized residue ' + name)
                return DataFrame([])
            N_xyz = residue['N'].get_vector()
            df = df.append({'aa': name, 'atom': 'N', 'res': res, 'coord': N_xyz}, ignore_index=True)
            CA_xyz = residue['CA'].get_vector()
            df = df.append({'aa': name, 'atom': 'CA', 'res': res, 'coord': CA_xyz}, ignore_index=True)
            C_xyz = residue['C'].get_vector()
            df = df.append({'aa': name, 'atom': 'C', 'res': res, 'coord': C_xyz}, ignore_index=True)
            res = res + 1

    bond_length = [(df.iloc[1].coord - df.iloc[0].coord).norm(), (df.iloc[2].coord - df.iloc[1].coord).norm()]
    bond_angle = [0, calc_angle(df.iloc[0].coord, df.iloc[1].coord, df.iloc[2].coord)*180/pi]
    torsion_angle = [0, 0]
    coord = [df.iloc[0].coord.get_array(), df.iloc[1].coord.get_array()]
    for ij in range(2, len(df)-1):
        bond_length.append((df.iloc[ij+1].coord - df.iloc[ij].coord).norm())
        bond_angle.append(calc_angle(df.iloc[ij-1].coord, df.iloc[ij].coord, df.iloc[ij+1].coord)*180/pi)
        torsion_angle.append(calc_dihedral(df.iloc[ij-2].coord, df.iloc[ij-1].coord, df.iloc[ij].coord, df.iloc[ij+1].coord)*180/pi)
        coord.append(df.iloc[ij].coord.get_array())
    bond_length.append(0)
    bond_angle.append(0)
    torsion_angle.append(0)
    coord.append(df.iloc[len(df)-1].coord.get_array())
    coord = array(coord)
    df_new = df.drop('coord', axis=1)
    df_new['x'] = coord[:, 0]
    df_new['y'] = coord[:, 1]
    df_new['z'] = coord[:, 2]
    df_new['bond_length'] = bond_length
    df_new['bond_angle'] = bond_angle
    df_new['torsion_angle'] = torsion_angle
    return df_new

def position(A, B, C, bc, R, theta, phi):
    n = cross(B-A, C-B)
    n = n/norm(n)
    D = array([R*cos(theta), R*sin(theta)*cos(phi), R*sin(theta)*sin(phi)])
    M = array([(C-B)/bc, cross(n, C-B)/bc, n]).T
    return matmul(M,D) + C

def reconstruct(init, R, bond_angle, torsion_angle):
    pos = init
    for ij in range(3, len(bond_angle)):
        pos.append(position(pos[ij-3], pos[ij-2], pos[ij-1], R[(ij-1)%3], R[ij%3], (180-bond_angle[ij-1])*pi/180, torsion_angle[ij-1]*pi/180))
    return array(pos)

def check(df):
    if df.bond_length[:-2].min() < 1:
        print('Short bond ' + str(df.bond_length[:-2].min()))
        return False
    if df.bond_length[:-2].max() > 2:
        print('Long bond ' + str(df.bond_length[:-2].max()))
        return False
    N_Ca = 1.458
    Ca_C = 1.525
    C_N = 1.329
    R = array([C_N, N_Ca, Ca_C])
    coord = stack([df.x, df.y, df.z], axis=1)
    init = [coord[0], coord[1], coord[2]]
    pos = reconstruct(init, R, df.bond_angle, df.torsion_angle)
    rmse = array([norm(x) for x in coord - pos]).mean()

    if rmse > 1:
        print('Large reconstruction error ' + str(rmse))
        return False

    return True

def parser_data(aa, path_to_pdbs, path_to_csv):

    if not os.path.exists(path=path_to_csv):
        os.makedirs(path_to_csv)

    if not os.path.exists(path_to_pdbs):
        raise ValueError("Path not exist {}".format(path_to_pdbs))

    files = glob.glob(path_to_pdbs + '*/*.pdb')

    p = PDBParser()

    for f in files:
        name = f[len(path_to_csv):]
        print(name)
        if not os.path.exists(path_to_csv + 'pdb-parsed' + name[6:-4] + '.csv') and not os.path.exists(path_to_csv + 'pdb-rejected' + name[6:-4] + '.csv'):
            try:

                structure = p.e('X', path_to_pdbs + name)
                df = parse(structure)
                flag = check(df)

                if not flag:
                    df = DataFrame([])

            except:
                df = DataFrame([])

            if len(df) > 0:
                print(name[6:-4] + ' success')
                df.to_csv(path_to_csv + 'pdb-parsed' + name[6:-4] + '.csv')
            else:
                df.to_csv(path_to_csv + 'pdb-rejected' + name[6:-4] + '.csv')

        else:
            print(name[7:-4] + ' already parsed or rejected')


if __name__ == '__main__':
    aa = ['PRO', 'TYR', 'THR', 'VAL', 'PHE', 'ARG', 'GLY', 'CYS', 'ALA',
          'LEU', 'MET', 'ASP', 'GLN', 'SER', 'TRP', 'LYS', 'GLU', 'ASN',
          'ILE', 'HIS']

    path_to_pdbs = "/mnt/"
    path_to_csv = "./PARSER_CSV/"

    parser_data(aa, path_to_pdbs, path_to_csv)

from src.Generator import PDBDataset
from src.utils import stack_pack, pad_packed_collate


aa = ['PRO', 'TYR', 'THR', 'VAL', 'PHE', 'ARG', 'GLY', 'CYS', 'ALA',
      'LEU', 'MET', 'ASP', 'GLN', 'SER', 'TRP', 'LYS', 'GLU', 'ASN',
      'ILE', 'HIS']
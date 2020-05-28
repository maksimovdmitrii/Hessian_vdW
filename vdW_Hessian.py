import numpy as np
from ase.io import read, write
from ase.constraints import Filter, FixAtoms
from ase.geometry.cell import cell_to_cellpar
from ase.optimize.precon.neighbors import estimate_nearest_neighbour_distance
from itertools import product

import optparse
parser = optparse.OptionParser()
parser.add_option("-f", "--formatfile", dest="formatfile", default='aims',
                    help="Input format extension ")
parser.add_option("-i", "--inputfile", dest="inputfile", help="input geometry file")
parser.add_option("-o", "--outputfile", dest="outputfile", help="output Hessian file")
#parser.add_option("-m", "--multiplier", dest="multiplier", help="vdW multiplier for C6 coeff")
options, args = parser.parse_args()


#    VALUES for ALPHA-vdW and C6_vdW:
#    !VVG: The majority of values such as isotropic static polarizability
#    !(in bohr^3), the homo-atomic van der Waals coefficient(in hartree*bohr^6),
#    !and vdW Radii (in bohr) for neutral free atoms are taken from Ref. Chu, X. & Dalgarno,
#    !J. Chem. Phys. 121, 4083 (2004) and Mitroy, et al.
#    !J. Phys. B: At. Mol. Opt. Phys. 43, 202001 (2010)
#    !and for rest of the elements they are calculated using linear response coupled cluster
#    !single double theory with accrate basis. The vdW radii for respective element are
#    !defined as discussed in Tkatchenko, A. & Scheffler, M. Phys. Rev. Lett. 102, 073005 (2009).

BOHR_to_angstr = 0.52917721   # in AA
HARTREE_to_eV = 27.211383  # in eV
HARTREE_to_kcal_mol = 627.509 # in kcal * mol^(-1)

# Ground state polarizabilities α0 (in atomic units) of noble gases and isoelectronic ions. 
# https://iopscience.iop.org/article/10.1088/0953-4075/43/20/202001/pdf
ALPHA_vdW = {'H': 4.5000, 'He': 1.3800, 'Li': 164.2000, 'Be': 38.0000, 'B': 21.0000, 'C': 12.0000,
                'N': 7.4000, 'O': 5.4000, 'F': 3.8000, 'Ne': 2.6700, 'Na': 162.7000, 'Mg': 71.0000, 'Al': 60.0000,
                'Si': 37.0000, 'P': 25.0000, 'S': 19.6000, 'Cl': 15.0000, 'Ar': 11.1000, 'K': 292.9000,
                'Ca': 160.0000,
                'Sc': 120.0000, 'Ti': 98.0000, 'V': 84.0000, 'Cr': 78.0000, 'Mn': 63.0000, 'Fe': 56.0000,
                'Co': 50.0000,
                'Ni': 48.0000, 'Cu': 42.0000, 'Zn': 40.0000, 'Ga': 60.0000, 'Ge': 41.0000, 'As': 29.0000,
                'Se': 25.0000,
                'Br': 20.0000, 'Kr': 16.8000, 'Rb': 319.2000, 'Sr': 199.0000, 'Y': 126.7370, 'Zr': 119.9700,
                'Nb': 101.6030,
                'Mo': 88.4225, 'Tc': 80.0830, 'Ru': 65.8950, 'Rh': 56.1000, 'Pd': 23.6800, 'Ag': 50.6000,
                'Cd': 39.7000,
                'In': 70.2200, 'Sn': 55.9500, 'Sb': 43.6719, 'Te': 37.65, 'I': 35.0000, 'Xe': 27.3000,
                'Cs': 427.12, 'Ba': 275.0,
                'La': 213.70, 'Ce': 204.7, 'Pr': 215.8, 'Nd': 208.4, 'Pm': 200.2, 'Sm': 192.1, 'Eu': 184.2,
                'Gd': 158.3, 'Tb': 169.5,
                'Dy': 164.64, 'Ho': 156.3, 'Er': 150.2, 'Tm': 144.3, 'Yb': 138.9, 'Lu': 137.2, 'Hf': 99.52,
                'Ta': 82.53,
                'W': 71.041, 'Re': 63.04, 'Os': 55.055, 'Ir': 42.51, 'Pt': 39.68, 'Au': 36.5, 'Hg': 33.9,
                'Tl': 69.92,
                'Pb': 61.8, 'Bi': 49.02, 'Po': 45.013, 'At': 38.93, 'Rn': 33.54, 'Fr': 317.8, 'Ra': 246.2,
                'Ac': 203.3,
                'Th': 217.0, 'Pa': 154.4, 'U': 127.8, 'Np': 150.5, 'Pu': 132.2, 'Am': 131.20, 'Cm': 143.6,
                'Bk': 125.3,
                'Cf': 121.5, 'Es': 117.5, 'Fm': 113.4, 'Md': 109.4, 'No': 105.4}

C6_vdW = {'H': 6.5000, 'He': 1.4600, 'Li': 1387.0000, 'Be': 214.0000, 'B': 99.5000, 'C': 46.6000,
            'N': 24.2000, 'O': 15.6000, 'F': 9.5200, 'Ne': 6.3800, 'Na': 1556.0000, 'Mg': 627.0000,
            'Al': 528.0000, 'Si': 305.0000, 'P': 185.0000, 'S': 134.0000, 'Cl': 94.6000, 'Ar': 64.3000,
            'K': 3897.0000, 'Ca': 2221.0000, 'Sc': 1383.0000, 'Ti': 1044.0000, 'V': 832.0000, 'Cr': 602.0000,
            'Mn': 552.0000, 'Fe': 482.0000, 'Co': 408.0000, 'Ni': 373.0000, 'Cu': 253.0000, 'Zn': 284.0000,
            'Ga': 498.0000, 'Ge': 354.0000, 'As': 246.0000, 'Se': 210.0000, 'Br': 162.0000, 'Kr': 129.6000,
            'Rb': 4691.0000, 'Sr': 3170.0000, 'Y': 1968.580, 'Zr': 1677.91, 'Nb': 1263.61, 'Mo': 1028.73,
            'Tc': 1390.87,
            'Ru': 609.754, 'Rh': 469.0, 'Pd': 157.5000, 'Ag': 339.0000, 'Cd': 452.0, 'In': 707.0460,
            'Sn': 587.4170,
            'Sb': 459.322, 'Te': 396.0, 'I': 385.0000, 'Xe': 285.9000, 'Cs': 6582.08, 'Ba': 5727.0, 'La': 3884.5,
            'Ce': 3708.33, 'Pr': 3911.84, 'Nd': 3908.75, 'Pm': 3847.68, 'Sm': 3708.69, 'Eu': 3511.71,
            'Gd': 2781.53, 'Tb': 3124.41, 'Dy': 2984.29, 'Ho': 2839.95, 'Er': 2724.12, 'Tm': 2576.78,
            'Yb': 2387.53, 'Lu': 2371.80, 'Hf': 1274.8, 'Ta': 1019.92, 'W': 847.93, 'Re': 710.2, 'Os': 596.67,
            'Ir': 359.1, 'Pt': 347.1, 'Au': 298.0, 'Hg': 392.0, 'Tl': 717.44, 'Pb': 697.0, 'Bi': 571.0,
            'Po': 530.92, 'At': 457.53, 'Rn': 390.63, 'Fr': 4224.44, 'Ra': 4851.32, 'Ac': 3604.41, 'Th': 4047.54,
            'Pa': 2367.42, 'U': 1877.10, 'Np': 2507.88, 'Pu': 2117.27, 'Am': 2110.98, 'Cm': 2403.22,
            'Bk': 1985.82,
            'Cf': 1891.92, 'Es': 1851.1, 'Fm': 1787.07, 'Md': 1701.0, 'No': 1578.18}

R0_vdW = {'H': 3.1000, 'He': 2.6500, 'Li': 4.1600, 'Be': 4.1700, 'B': 3.8900, 'C': 3.5900,
            'N': 3.3400, 'O': 3.1900, 'F': 3.0400, 'Ne': 2.9100, 'Na': 3.7300, 'Mg': 4.2700,
            'Al': 4.3300, 'Si': 4.2000, 'P': 4.0100, 'S': 3.8600, 'Cl': 3.7100, 'Ar': 3.5500,
            'K': 3.7100, 'Ca': 4.6500, 'Sc': 4.5900, 'Ti': 4.5100, 'V': 4.4400, 'Cr': 3.9900,
            'Mn': 3.9700, 'Fe': 4.2300, 'Co': 4.1800, 'Ni': 3.8200, 'Cu': 3.7600, 'Zn': 4.0200,
            'Ga': 4.1900, 'Ge': 4.2000, 'As': 4.1100, 'Se': 4.0400, 'Br': 3.9300, 'Kr': 3.8200,
            'Rb': 3.7200, 'Sr': 4.5400, 'Y': 4.8151, 'Zr': 4.53, 'Nb': 4.2365, 'Mo': 4.099,
            'Tc': 4.076, 'Ru': 3.9953, 'Rh': 3.95, 'Pd': 3.6600, 'Ag': 3.8200, 'Cd': 3.99,
            'In': 4.2319, 'Sn': 4.3030, 'Sb': 4.2760, 'Te': 4.22, 'I': 4.1700, 'Xe': 4.0800,
            'Cs': 3.78, 'Ba': 4.77, 'La': 3.14, 'Ce': 3.26, 'Pr': 3.28, 'Nd': 3.3,
            'Pm': 3.27, 'Sm': 3.32, 'Eu': 3.40, 'Gd': 3.62, 'Tb': 3.42, 'Dy': 3.26,
            'Ho': 3.24, 'Er': 3.30, 'Tm': 3.26, 'Yb': 3.22, 'Lu': 3.20, 'Hf': 4.21,
            'Ta': 4.15, 'W': 4.08, 'Re': 4.02, 'Os': 3.84, 'Ir': 4.00, 'Pt': 3.92,
            'Au': 3.86, 'Hg': 3.98, 'Tl': 3.91, 'Pb': 4.31, 'Bi': 4.32, 'Po': 4.097,
            'At': 4.07, 'Rn': 4.23, 'Fr': 3.90, 'Ra': 4.98, 'Ac': 2.75, 'Th': 2.85,
            'Pa': 2.71, 'U': 3.00, 'Np': 3.28, 'Pu': 3.45, 'Am': 3.51, 'Cm': 3.47,
            'Bk': 3.56, 'Cf': 3.55, 'Es': 3.76, 'Fm': 3.89, 'Md': 3.93, 'No': 3.78}


def vector_separation(cell_h, cell_ih, qi, qj):
    # This file is part of i-PI.
    # i-PI Copyright (C) 2014-2015 i-PI developers
    # See the "licenses" directory for full license information.


    """Calculates the vector separating two atoms.

       Note that minimum image convention is used, so only the image of
       atom j that is the shortest distance from atom i is considered.

       Also note that while this may not work if the simulation
       box is highly skewed from orthorhombic, as
       in this case it is possible to return a distance less than the
       nearest neighbour distance. However, this will not be of
       importance unless the cut-off radius is more than half the
       width of the shortest face-face distance of the simulation box,
       which should never be the case.

       Args:
          cell_h: The simulation box cell vector matrix.
          cell_ih: The inverse of the simulation box cell vector matrix.
          qi: The position vector of atom i.
          qj: The position vectors of one or many atoms j shaped as (N, 3).
       Returns:
          dij: The vectors separating atoms i and {j}.
          rij: The distances between atoms i and {j}.
    """

    sij = np.dot(cell_ih, (qi - qj).T)  # column vectors needed
    sij -= np.rint(sij)

    dij = np.dot(cell_h, sij).T         # back to i-pi shape
    rij = np.linalg.norm(dij, axis=1)

    return dij, rij


atoms = read(options.inputfile, format = options.formatfile)

N  = len(atoms)
coordinates = atoms.get_positions()
atom_names = atoms.get_chemical_symbols()
cell_h = atoms.get_cell()[:]
cell_ih = atoms.get_reciprocal_cell()[:]

def calculate_vdW(i, j, rij, coordinates, atom_names):
    
    # i - one Index
    # j - many indeces

    def calculate_vdw_block(i_ind, j_ind, coord, dist, C6_coeff):
        pairs = product(range(3), repeat=2)
        block = np.array([(coord[i_ind][k[0]] - coord[j_ind][k[0]])
                 * (coord[i_ind][k[1]] - coord[j_ind][k[1]]) for k in pairs])
        return C6_coeff * ((-48 / dist ** 10) + (168 / dist ** 16)) * block 

    

    # C6 coefficient for atoms A and B
    C6i = C6_vdW[atom_names[i]]
    C6j = [C6_vdW[atom_names[a]] for a in j]     
    
    # polarizabilities α
    alphai = ALPHA_vdW[atom_names[i]]  
    alphaj = [ALPHA_vdW[atom_names[a]]  for a in j]
    
      
    units = HARTREE_to_eV * (BOHR_to_angstr ** 6)
    
    C6AB = [(2 * C6i * C6j[z]) 
            / (alphaj[z] / alphai * C6i
                + alphai / alphaj[z] * C6j[z]) 
            * units
            for z in range(len(j))]

    blocks = [(np.identity(3).reshape(1, -1) * C6AB[n] *
                    (6 / (rij[n] ** 8) - 12 * 1 / (rij[n] ** 14))).reshape(3, 3)
                     + (calculate_vdw_block(i, j[n], coordinates, rij[n], C6AB[n])).reshape(3, 3)
                     for n in range(len(j))]

    return np.hstack(blocks)
       

hessian = np.zeros(shape = (3 * N, 3 * N))
for i in range(N-1):
    qi = coordinates[i].reshape(1,3)
    qj = coordinates[i+1:].reshape(-1,3)
    
    if np.array_equal(cell_h,np.zeros([3, 3])):
        rij = np.array([np.linalg.norm(qi-Qj) for Qj in qj])
    else:
        dij, rij = vector_separation(cell_h, cell_ih, qi, qj) 
    
    j = range(N)[i+1:]
    stack = calculate_vdW(i, j, rij, coordinates, atom_names)
    hessian[3 * i    , 3 * (i+1):] = stack[0]
    hessian[3 * i + 1, 3 * (i+1):] = stack[1]
    hessian[3 * i + 2, 3 * (i+1):] = stack[2]
    
# Fill lower triangle
hessian = hessian + hessian.T - np.diag(hessian.diagonal())

# Regularization
for ind in range(len(hessian)):
    hessian[ind, ind] = hessian[ind, ind] + 0.5

with open(options.outputfile, 'w') as hes:
    for i in hessian:
        hes.write(' '.join(["{:8.3f}".format(k) for k in i]))
        hes.write('\n')

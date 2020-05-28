import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
from ase.optimize.precon import Exp, PreconLBFGS
from ase.io import read, write
from ase.optimize import BFGS
import os
import pandas as pd
from ase.calculators.lj import LennardJones

from ase.calculators.lammpslib import LAMMPSlib


from ase.calculators.loggingcalc import LoggingCalculator
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

import optparse
parser = optparse.OptionParser()
parser.add_option("-f", "--formatfile", dest="formatfile", default='aims',
                    help="Input format extension ")
parser.add_option("-i", "--inputfile", dest="inputfile", help="input geometry file")
parser.add_option("--hessian", dest="hessian", help="Preconditionered Hessian file")
parser.add_option("-o", "--outputfile", dest="outputfile", help="output Trajectory file")
options, args = parser.parse_args()

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

#a0 = bulk('Cu', cubic=True)
#a0 *= [3, 3, 3]
#del a0[0]
#a0.rattle(0.1)
a0 = read(options.inputfile, format = options.formatfile)

symbol = a0.get_chemical_symbols()[0]

ABOHR = 0.52917721 # in AA
HARTREE = 27.211383 # in eV

#epsilon = 0.185 # in kcal * mol^(-1)
#epsilon = 0.000294816 # in Hartree
#epsilon = 0.008022361 # in eV

#sigma = R0_vdW[symbol]
epsilon = 0.0103
sigma = 3.4

lammps_header=[
                "dimension     3",
                "boundary      p p p",
                "atom_style    atomic",
                "units         metal",
                #"neighbor      2.0  bin",
                'atom_modify     map array',
                ]
lammps_cmds = [
                'pair_style lj/cut 15.0',
                'pair_coeff * *  {} {}'.format(epsilon ,sigma),
                #'pair_coeff 1 1  0.238 3.405',
                #'fix         1 all nve',
                ]
#atom_types={'Ar':1}
lammps = LAMMPSlib(lmpcmds=lammps_cmds,
            #atom_types=atom_types, 
            lammps_header=lammps_header,
            log_file='LOG.log', 
            keep_alive=True)    


#calculator = LennardJones()   
calculator = lammps       
atoms = a0.copy()
atoms.set_calculator(calculator)

opt = BFGS(atoms, trajectory=options.outputfile)
#opt = BFGS(atoms)
opt.H0 = pd.read_csv(options.hessian, sep='\s+', header=None)
opt.run(fmax=1e-3, steps=300)
xyz_out = options.outputfile.replace(".traj", ".xyz").replace("Trajectories_vdW", "xyz_vdW")
write(xyz_out, atoms, format="xyz")

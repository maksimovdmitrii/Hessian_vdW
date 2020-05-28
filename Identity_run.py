import numpy as np
from ase.build import bulk
from ase.calculators.emt import EMT
#from ase.optimize.precon import vdW, Exp, PreconLBFGS
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
parser.add_option("-o", "--outputfile", dest="outputfile", help="output Trajectory file")
#parser.add_option("-m", "--multiplier", dest="multiplier", help="vdW multiplier for C6 coeff")
options, args = parser.parse_args()

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
#opt.H0 = hessian
opt.run(fmax=1e-3, steps=300)

xyz_out = options.outputfile.replace(".traj", ".xyz").replace("Trajectories_ID", "xyz_ID")
write(xyz_out, atoms, format="xyz")

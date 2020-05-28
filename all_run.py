import os, sys, re, shutil
import numpy as np
import gc

from ase.io import read, write
from ase.calculators.lj import LennardJones
from ase.calculators.lammpslib import LAMMPSlib


# Load Directories
xyzDir = os.path.join(os.getcwd(), "xyz")
inDir = os.path.join(os.getcwd(), "in")
rawDir = os.path.join(os.getcwd(), "raw")
hesDir = os.path.join(os.getcwd(), "Hessians")   
trajvdWDir = os.path.join(os.getcwd(), "Trajectories_vdW")
trajIDDir = os.path.join(os.getcwd(), "Trajectories_ID")
picDir = os.path.join(os.getcwd(), "Pictures")

# Run all structures
for i in sorted(os.listdir(inDir)):
    name = i.split(".")[0]
    inputfile = os.path.join(inDir, name+".in")
    hessian = os.path.join(hesDir, name+".hes")
    
    gc.collect()
    outputfile = os.path.join(trajIDDir, name+".traj")
    os.system("python Identity_run.py -i {} -o {}".format(inputfile, outputfile))
    
    outputfile = os.path.join(trajvdWDir, name+".traj")
    os.system("python precon_vdW_run.py -i {} --hessian {} -o {}".format(inputfile, hessian, outputfile))
    gc.collect()
   
    

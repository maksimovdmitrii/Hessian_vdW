import os, sys, re, shutil
import numpy as np

from ase.io import read, write
from ase.calculators.lj import LennardJones
from ase.calculators.lammpslib import LAMMPSlib


# Create Directories
xyzDir = os.path.join(os.getcwd(), "xyz")
if not os.path.exists(xyzDir):
    os.mkdir(xyzDir)

inDir = os.path.join(os.getcwd(), "in")
if not os.path.exists(inDir):
    os.mkdir(inDir)

rawDir = os.path.join(os.getcwd(), "raw")

hesDir = os.path.join(os.getcwd(), "Hessians")
if not os.path.exists(hesDir):
    os.mkdir(hesDir)
    
trajvdWDir = os.path.join(os.getcwd(), "Trajectories_vdW")
if not os.path.exists(trajvdWDir):
    os.mkdir(trajvdWDir)

trajIDDir = os.path.join(os.getcwd(), "Trajectories_ID")
if not os.path.exists(trajIDDir):
    os.mkdir(trajIDDir)
    
picDir = os.path.join(os.getcwd(), "Pictures")
if not os.path.exists(picDir):
    os.mkdir(picDir)

xyzIDDir = os.path.join(os.getcwd(), "xyz_ID")
if not os.path.exists(xyzIDDir):
    os.mkdir(xyzIDDir) 
    
xyzvdWDir = os.path.join(os.getcwd(), "xyz_vdW")
if not os.path.exists(xyzvdWDir):
    os.mkdir(xyzvdWDir) 
    
# Produce XYZ files
# That are in r0 coordinates

# For Ar 
epsilon = 0.0103
sigma = 3.4

for i in os.listdir(rawDir):
    with open(os.path.join(xyzDir, "{:04d}.xyz".format(int(i))), "w") as xyz:
        lines = open(os.path.join(rawDir, i)).readlines()
        lenAtoms = len(lines)
        xyz.write("{}\n\n".format(lenAtoms))
        for line in lines:
            coords = np.array([float(z) for z in line.split()]) * sigma
            xyz.write("Ar {}\n".format("   ".join([str(s) for s in coords])))

# Produce FHI-aims files
for i in os.listdir(xyzDir):
    struc = read(os.path.join(xyzDir, i), format="xyz")
    struc.rattle(0.05)
    struc.set_cell(np.eye(3) * 50)
    struc.set_pbc(np.ones(3) * 50)
    #print(np.eye(3) * 50)
    #import sys
    #sys.exit(0)
    write(os.path.join(inDir, "{}.in".format(i.split('.')[0])), struc, format="aims")
    
# Produce Hessians
for structure in sorted(os.listdir(inDir)):
    inputfile = os.path.join(inDir, structure)
    outputfile = os.path.join(hesDir, structure.split(".")[0]+".hes")
    os.system("python vdW_Hessian.py -i {} -o {} -f aims".format(inputfile, outputfile))
    print("Hessian for {} is done".format(structure))
    




import os, sys, re
import numpy as np
from ase.io import read, Trajectory
from optparse import OptionParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from ase.visualize.plot import plot_atoms
from matplotlib import rc, rcParams
import matplotlib.font_manager
pgf_with_rc_fonts = {"pgf.texsystem": "pdflatex"}
matplotlib.rcParams.update(pgf_with_rc_fonts)
#rc('text', usetex = True)
rcParams['mathtext.fontset'] = 'cm'
rcParams['mathtext.it'] = 'Arial:italic'
rcParams['mathtext.rm'] = 'Arial'

SMALL_SIZE = 14
MEDIUM_SIZE = 14.5
BIGGER_SIZE = 16
plt.rc('font', size=MEDIUM_SIZE)  # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)  # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE) # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)  # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
#plt.rcParams['font.family'] = 'sans-serif'
#plt.rcParams['font.sans-serif'] = ['Tex Gyre Heros'] + plt.rcParams['font.sans-serif']
#plt.rcParams['mathtext.fontset'] = 'dejavuserif'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'

from optparse import OptionParser
parser = OptionParser()
parser.add_option("-r", "--run", dest="run", help="Run the test")
parser.add_option("-a", "--analyze", dest="analyze", help="Analyze the run")
(options, args) = parser.parse_args()


# Load Directories
xyzDir = os.path.join(os.getcwd(), "xyz")
inDir = os.path.join(os.getcwd(), "in")
rawDir = os.path.join(os.getcwd(), "raw")
hesDir = os.path.join(os.getcwd(), "Hessians")   
trajvdWDir = os.path.join(os.getcwd(), "Trajectories_vdW")
trajIDDir = os.path.join(os.getcwd(), "Trajectories_ID")
picDir = os.path.join(os.getcwd(), "Pictures")

def get_cartesian_rms(atoms1, atoms2):
    
    """Return the optimal RMS after aligning two structures."""
    atoms1.set_positions(atoms1.get_positions() - atoms1.get_center_of_mass())
    atoms2.set_positions(atoms2.get_positions() - atoms2.get_center_of_mass())
    '''Kabsh'''
    A = np.dot(atoms1.get_positions().T, atoms2.get_positions())
    V, S, W = np.linalg.svd(A)
    if np.linalg.det(np.dot(V, W)) < 0.0:
        V[:, -1] = -V[:, -1]
        K = np.dot(V, W)
    else:
        K = np.dot(V, W)

    atoms1.set_positions(np.dot(atoms1.get_positions(), K))
    rmsd_kabsh = 0.0
    for v, w in zip(atoms1.get_positions(), atoms2.get_positions()):
        rmsd_kabsh += sum([(v[i] - w[i])**2.0 for i in range(len(atoms1.get_positions()[0]))])

    return np.sqrt(rmsd_kabsh/len(atoms1.get_positions()))



names = [k.split(".")[0] for k in os.listdir(trajvdWDir)]
fig= plt.figure(figsize=(4, 4))
RMSD = [get_cartesian_rms(
        Trajectory(os.path.join(trajIDDir, name+".traj"))[-1], 
        Trajectory(os.path.join(trajvdWDir, name+".traj"))[-1]) 
        for name in sorted(names)]


performance = [len(Trajectory(os.path.join(trajIDDir, name+".traj")))
               /len(Trajectory(os.path.join(trajvdWDir, name+".traj")))
                for name in sorted(names)] 
                #if len(Trajectory(os.path.join(trajvdWDir, name+".traj"))) < 300]

y_pos = np.arange(len(performance)) 
for i,k in zip(RMSD, performance):
    print(round(i, 2), round(k, 2))


#performance = [len(Trajectories[0])/float(len(i)) if len(i)<2000 else 1000 for i in Trajectories]
colors=[]
#for x in performance:
    #if x==max(performance):
        #colors.append('#45FF42')
    #elif x==min(performance):
        #colors.append('#CC3D33')
    #else:
        #colors.append('#7872A3')


for x in performance:
    if x==max(performance):
        colors.append('#7872A3')
    elif x==min(performance):
        colors.append('#7872A3')
    else:
        colors.append('#7872A3')
        
#for i in range(len(forces)):
    #performance.append(float(len(identity_forces))/len(forces[i]))
    #objects.append('Lindh_vdW_Exp_{}_{}'.format(str(mu_alpha[i][0]), str(mu_alpha[i][1])))

plt.bar(y_pos, performance, width=.75, color=colors, align='center', alpha=0.75)
#plt.bar(y_pos, RMSD, width=.75,  align='center', alpha=0.75, color=["#7872A3" for i in RMSD])
#plt.bar(y_pos, performance, width=.75, color=colors, align='center', alpha=0.75)
#plt.xticks(y_pos, objects, rotation=0)

#tickmnames = [str(i) for i in sorted(names) if int(i)%10==0]
#tickpositions = [int(i) * 3  for i in tickmnames]

#plt.xticks(tickmnames, tickpositions)
plt.xlim(min(y_pos), max(y_pos))
plt.ylim(0, 12)
#plt.hlines(min(performance), min(y_pos)-2, max(y_pos)+2, color='k', linestyle='--')
plt.hlines(1, min(y_pos), max(y_pos), color='k', linestyle='--')
plt.hlines(np.average(performance), min(y_pos), max(y_pos), color='#CC3D33', linestyle='--')
plt.ylabel('Performance gain')
plt.xlabel('Cluster size')
#plt.title('Comparison of vdW\nPreconditioned Hessians\nfor different Ar (3-150) clusters')
plt.title('Different Ar (3-150) clusters')
#fig.autofmt_xdate()
plt.tight_layout()
plt.savefig(os.path.join(picDir, "vdW_ID.png"), dpi=300)
#plt.show()








#hessians = {
    #"python Exp_Hessian.py"   : "-g geometry.in --A 3",
    #"python2.7 Lindh_Hessian.py" : "geometry.in --full Hessian_Lindh.dat",
    #"python vdW_Hessian.py"   : "-g geometry.in -m 1",
    #}

#hesDir = os.path.join(os.getcwd(), "Hessians")

#for i in hessians:
    #os.system("cd {} && {} {}".format(hesDir, i, hessians[i]))
    #print("Done {}".format(i))
    



#run = [
       #"Identity_run.py", 
       #"precon_Exp_run.py",
       #"precon_Lindh_run.py",
       #"precon_vdW_run.py" 
       #]

#for i in run:
    #os.system("python {}".format(i))
    #print("Done {}".format(i))

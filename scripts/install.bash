#The following steps outline how to install the necessary libraries and dependencies on the Gadi supercomputer at NCI, Canberra:

cd ~
module load python3/3.11.0
module load openmpi/4.0.5
python3 -m venv env_MPI --system-site-package
source ~/env_MPI/bin/activate
python3 -m pip install mpi4py mgmetis jupyter
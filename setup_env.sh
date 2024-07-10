source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
module load python3/3.11.0
module load openmpi/4.0.5

export n_meshparts=24
export work_dir=~/PCG_MPI
cd $work_dir

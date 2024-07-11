## Project description


## Environment Setup

```shell
$cd ~
$module load python3/3.11.0
$module load openmpi/4.0.5
$python3 -m venv env_MPI --system-site-package
$source ~/env_MPI/bin/activate
$python3 -m pip install mpi4py mgmetis jupyter
```

## Run jupyter notebook from terminal

```shell
module load python3/3.11.0
source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
$cd ~/PCG_MPI
$jupyter nbconvert --execute --clear-output solver_demo.ipynb
```

## Project description
This is an mpi-based parallel pcg solver.

## Installation
To install this project, follow these steps:

```shell
cd ~
module load python3/3.11.0
module load openmpi/4.0.5
python3 -m venv env_MPI --system-site-package
source ~/env_MPI/bin/activate
python3 -m pip install mpi4py mgmetis jupyter
```

## Usage
To use this project, first setup the environment:
```shell
module load python3/3.11.0
source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
cd ~/PCG_MPI_Solver
```
Then, to test run, you can choose one of the following options
#### 1. Run bash file
```shell
bash run_script.bash
```

#### 2. Run jupyter notebook
```shell
jupyter nbconvert --execute --clear-output solver_demo.ipynb
```

## Citation
If you use this code in a scientific publication, please cite the following paper:

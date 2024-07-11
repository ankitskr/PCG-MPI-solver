## Project description
This is an mpi-based parallel pcg solver.

## Installation
To setup this project on a supercomputing cluster, follow these steps:
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
```

Save the repository "PCG_MPI_Solver" in the home folder of desktop system or supercomputing cluster.
```shell
cd ~/PCG_MPI_Solver
```

Then, execute the bash file to test run the solver.
```shell
bash scripts/run_script.bash
```

## Working demonstration

A working demonstration for an example problem is provided in a jupyter notebook.
Refer to notebooks/solver_demo.ipynb

## Documentation


## Citation
If you use this code in a scientific publication, please cite the following paper:

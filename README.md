# PCG-MPI-solver

## Project Description
This is an MPI-based parallel Preconditioned Conjugate Gradient (PCG) solver. The solver has been tested on the Gadi supercomputer using approximately 12,000 cores for solving linear elastostatic problems with over 1 billion unknowns (dofs).

The PCG is an algorithm for the numerical solution of systems of linear equations (**Ax=b**) with positive-semidefinite system matrix (**A**). The method can also be used to solve unconstrained optimization problems such as energy minimization.

The Message Passing Interface (MPI) is a standardized and portable message-passing system, designed for developing efficient parallel codes on distributed computing architectures.

## Installation

### On Gadi Supercomputer
The following steps outline how to install the necessary libraries and dependencies on the Gadi supercomputer at NCI, Canberra:

```shell
cd ~
module load python3/3.11.0
module load openmpi/4.0.5
python3 -m venv env_MPI --system-site-package
source ~/env_MPI/bin/activate
python3 -m pip install mpi4py mgmetis jupyter
```

### On a Local Workstation
To run the code on a local workstation, ensure you have the following installed:

- Python 3
- Open-MPI
- Python libraries: numpy, scipy, matplotlib, Cython, mpi4py, mgmetis, jupyter

You can install the required Python libraries using pip:

```shell
pip install numpy scipy matplotlib Cython mpi4py mgmetis jupyter
```


## Usage

1. Save the repository "PCG_MPI_Solver" in your home directory and set it as the working directory:
```shell
cd ~/PCG_MPI_Solver
```

2. Execute the bash file to set up the environment and test run the solver:
```shell
bash examples/run_basic_script.bash
```

3. Alternatively, setup the environment as follows and run the jupyter notebook.
```shell
module load python3/3.11.0
source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
jupyter nbconvert --execute --clear-output notebooks/solver_demo.ipynb
```

## Working demonstration
A working demonstration of this code for an example problem is provided in the jupyter notebook.
Refer to [notebooks/solver_demo.ipynb](https://github.com/ankitskr/PCG-MPI-solver/blob/main/notebooks/solver_demo.ipynb)



## Project structure
Below is the folder structure of the PCG_MPI_Solver project, outlining the main directories and their contents:
```
PCG_MPI_Solver/
├── .gitignore                  
├── LICENSE                     # Contains the licensing agreement for the project
├── README.md                   # Provides a detailed description of the project
├── data/                       
│   └── concrete.zip            # Model data used by the project for demonstration
├── docs/
│   └── references.txt          # Documentation and references related to the project
├── examples/
│   └── run_basic_script.bash   # Bash script for running a basic example of the solver
├── notebooks/
│   ├── setup_env.sh            # Shell script for setting up the environment
│   ├── solver_demo.ipynb       # Jupyter notebook demonstrating the solver's usage
│   └── images/                 # Directory containing images used in notebooks
├── scripts/
│   └── install.bash            # Script to automate the installation of dependencies
└── src/
    ├── solver/                 # Core directory for the solver's algorithms
    │   ├── partition_mesh.py   # Script for mesh partitioning
    │   ├── pcg_solver.py       # Script for the preconditioned conjugate gradient solver
    │   └── run_metis.py        # Script to run METIS for mesh partitioning
    ├── utils/                  # Utility scripts for the project
    │   └── file_operations.py  # Script to handle file operations
    └── data/
        ├── export_vtk.py       # Script to export data in VTK format
        ├── read_input_model.py # Script to read the input model from files
        └── evtk/               # EVTK library to convert data in VTK format
```


## Documentation
1. For detailed information on the numerical techniques and solver algorithms employed in this project, refer to the article:

- **Title:** "An octree pattern-based massively parallel PCG solver for elasto-static and dynamic problems"
- **Link:** [Article - Computer Methods in Applied Mechanics and Engineering](https://doi.org/10.1016/j.cma.2022.115779)

2. Additional details can be found in the PhD thesis:

- **Title:** "High-Performance Computing for Impact-Induced Fracture Analysis exploiting Octree Mesh Patterns"
- **Link:** [PhD Thesis - UNSW Sydney](https://doi.org/10.26190/unsworks/22788)


## Citation
If you use this code in a scientific publication, please cite the following paper:

Ankit A, Zhang J, Eisenträger S, Song C. An octree pattern-based massively 
parallel PCG solver for elasto-static and dynamic problems. Computer Methods 
in Applied Mechanics and Engineering. 2023 Feb 1;404:115779.
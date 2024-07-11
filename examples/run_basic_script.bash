#Setup environment
source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
module load python3/3.11.0
module load openmpi/4.0.5

export work_dir=~/PCG_MPI_Solver
cd $work_dir

export n_meshparts=8


#Load model
export model_name=concrete
export scratch_path=$work_dir/data
export input_model_path=$scratch_path/$model_name.zip
python3 src/data/read_input_model.py $work_dir $model_name $scratch_path $input_model_path


#Generate indices for model partitioning
python3 src/solver/run_metis.py $n_meshparts

   
#Partitioning model
export n_cores=4   
mpiexec -np $n_cores --map-by numa python3 src/solver/partition_mesh.py $n_meshparts 0


#Defining solver parameters
python3 -c "
from src.utils.file_operations import exportz

#Time-history settings
TimeHistoryParam={'ExportFlag': True, # export results
                  'ExportFrmRate' : 1,
                  'ExportFrms' : [],
                  'PlotFlag' : False,
                  'TimeStepDelta': [0,1],
                  'ExportVars': 'U'} #D U PS PE GS GE

#PCG solver parameters
SolverParam={'Tol' : 1e-7,
             'MaxIter' : 10000}

#Exporting settings
GlobSettings = {'TimeHistoryParam':TimeHistoryParam,
                'SolverParam':SolverParam}
exportz('__pycache__/GlobSettings.zpkl', GlobSettings)
"

#Running PCG solver
mpiexec -np $n_meshparts --map-by numa python3 src/solver/pcg_solver.py 1 0

#Exporting results for visualisation
mpiexec -np 1 python3 src/data/export_vtk.py 1 "U" "Full"

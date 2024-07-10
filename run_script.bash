
source ~/env_MPI/bin/activate
export PYTHONPATH=~/env_MPI/lib/python3.11/site-packages:$PYTHONPATH
module load python3/3.11.0
module load openmpi/4.0.5


export n_meshparts=8
export work_dir=~/PCG_MPI
cd $work_dir


export model_name=Concrete
export scratch_path=$work_dir/$model_name
export input_model_path=$scratch_path/$model_name.zip
python3 read_input_model.py $work_dir $model_name $scratch_path $input_model_path


python3 run_metis.py $n_meshparts

   
export n_cores=4   
mpiexec -np $n_cores --map-by numa python3 partition_mesh.py $n_meshparts 0


python3 -c "
from file_operations import exportz

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

mpiexec -np $n_meshparts --map-by numa python3 pcg_mpi_solver.py 1 0


mpiexec -np 1 python3 export_vtk.py 1 "U" "Full"

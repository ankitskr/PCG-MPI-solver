
source ~/env/bin/activate
export PYTHONPATH=~/env/lib/python3.7/site-packages:$PYTHONPATH
module load python3/3.7.4
module load openmpi/4.0.7

export NSplits=4
export NPrt=12
export NMetisParts=1
export Model=CubeTest
export SolverFile=Elast_StaticAnalysis.py
export WorkDir=~/MatlabLink_4Jul/High_Performance_Computing_MPI/
cd $WorkDir


mpiexec -np $NMetisParts --map-by numa python3 MetisMeshPart.py $WorkDir
    
mpiexec -np $NSplits --map-by numa python3 Elast_ParDataPrepMPI.py $NPrt $WorkDir 0

mpiexec -np $NPrt --map-by numa python3 $SolverFile $Model $WorkDir 1 0 0

mpiexec -np 1 python3 ExportVTK.py $Model $WorkDir 1 "U" "Full"

mpiexec -np 1 python3 ExportMat.py $Model $WorkDir 1 "U"
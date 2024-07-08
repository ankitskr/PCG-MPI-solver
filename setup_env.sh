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

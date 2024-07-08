export ModelName=Foam_high_Eigen
NCPUS=(       48   48   48   48   48   48   96  144 288  528 1056 2064  4128)
TotMemList=( 190  190  190  190  190  190  360  540 960 1800 3600 7200 14400)
#TotMemList=(1000 1000 1000 1000 1000 1000 2000 3000)
#		           0    1    2    3    4    5    6    7   8    9   10   11    12
#export Queue=hugemem
export Queue=normal
export WallTime=1:00:00

export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver/
cd ${WorkDir}PythonMPI/

export R0=2

for P0 in 10; do 
	
	  export P0=$P0
    export JobName=${ModelName}_N$P0
	  export NCPU=${NCPUS[$P0]}
    export TotMem=${TotMemList[$P0]}GB
	  
    qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v ModelName,WorkDir,ScratchPath,P0,R0 Eigen_Analysis.pbs;
	
	
    sleep 2;

done
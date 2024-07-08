export ModelName=PureBending_Octree_0
NCPUS=(      48  48  48  48  48  48  96 144 288  528 1056 2064  4128)
TotMemList=(190 190 190 190 190 190 380 540 960 1800 3600 7200 14400)
#		         0    1   2   3   4   5   6   7   8    9   10   11    12
#export Queue=hugemem
export Queue=normal
export WallTime=1:00:00

export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver/
cd ${WorkDir}PythonMPI/


for P0 in 3 4 5; do 
	
	  export P0=$P0
    export JobName=${ModelName}_N$P0
	  export NCPU=${NCPUS[$P0]}
    export TotMem=${TotMemList[$P0]}GB
	  
    qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v ModelName,WorkDir,ScratchPath,P0 PCG_Analysis.pbs;
	
	
    sleep 5;

done
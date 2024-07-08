export ModelName=CONCRETE_AVIZO_256
NCPUS=(       48   48   48   48   48   48   48   96  192  384   786   1536)
N_MshPrts=(    1    2    3    6   12   24   48   96  192  384   768   1536)
TotMemList=( 190  190  190  190  190  190  190  380  760 1560  3120   6240)
#TotMemList=(1000 1000 1000 1000 1000 1000 1000 2000)
#		       0    1    2    3    4    5    6    7    8    9    10     11
#export Queue=hugemem
export Queue=normal
export WallTime=4:00:00

export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver_Contact/
cd ${WorkDir}PythonMPI/

export SpeedTestFlag=1 #0 or 1
export R0=3

for P0 in 0 1; do 
	
	  export N_MshPrt=${N_MshPrts[$P0]}
    export JobName=${ModelName}_N$P0
	  export NCPU=${NCPUS[$P0]}
    export TotMem=${TotMemList[$P0]}GB
	  
    qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v ModelName,WorkDir,ScratchPath,N_MshPrt,R0,SpeedTestFlag PCGEff_Analysis.pbs;
	
	
    sleep 5;

done
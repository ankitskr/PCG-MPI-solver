export ModelName=DynCoh_SandwichPanel_2x2
NCPUS=(        48   48   48   48   48   48   48    96   192   384    768   1536    3072   6144   12288) # 20736
N_MshPrts=(     1    2    3    6   12   24   48    96   192   384    768   1536    3072   6144   12288)
TotMemList=(  190  190  190  190  190  190  190   380   760  1520   3040   6080   12160  24320   48640)
#TotMemList=(1500 1500 1500 1500 1500 1500 1500  3000  6000)
#		        0    1    2    3    4    5    6     7     8     9     10     11      12     13      14
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
export R0=6

for P0 in 9 10 11; do
	
	  export N_MshPrt=${N_MshPrts[$P0]}
    export JobName=${ModelName}_N$N_MshPrt
	  export NCPU=${NCPUS[$P0]}
    export TotMem=${TotMemList[$P0]}GB
	  
    qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v ModelName,WorkDir,ScratchPath,N_MshPrt,R0,SpeedTestFlag DynCohesive_Analysis.pbs;
	
	
    sleep 5;

done

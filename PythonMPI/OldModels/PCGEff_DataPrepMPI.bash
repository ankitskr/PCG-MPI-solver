export ModelName=CONCRETE_AVIZO_256
export NCPU=4
export TotMem=120GB
export Queue=hugemem
export WallTime=1:00:00


export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver_Contact/
cd ${WorkDir}PythonMPI/
N_MshPrts=(    1    2    3    6   12   24   48   96  192  384   768   1536)
#		           0    1    2    3    4    5    6    7    8    9    10     11


for P0 in 0; do 

	  export N_MshPrt=${N_MshPrts[$P0]}
    export JobName=Msh_${ModelName}_N$P0
    
	  qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v N_MshPrt,WorkDir,ScratchPath PCGEff_DataPrepMPI.pbs; 
    
    sleep 2;

done
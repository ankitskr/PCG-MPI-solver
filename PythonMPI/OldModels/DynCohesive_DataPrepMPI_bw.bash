export ModelName=DynCoh_SandwichPanel_4x4
export NCPU=32 #1 (for serial); or 32 (for parallel)
export dN=0.25 #1.0, 0.5, 0.25 (for parallel only: int(dN*32) must be multiple of 4 -- A lower value of dN reduces memory requirement but delays parallelization)
export TotMem=3000GB
export JobFS=400GB
export Queue=megamembw
export WallTime=12:00:00


export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver_Contact/
cd ${WorkDir}PythonMPI/
N_MshPrts=(    1    2    3    6   12   24   48   96  192  384   768   1536   3072   6144  12288  20736)
#		       0    1    2    3    4    5    6    7    8    9    10     11     12     13     14     15


for P0 in 13 14; do 
	
	  export N_MshPrt=${N_MshPrts[$P0]}
    export JobName=Msh_${ModelName}_N$N_MshPrt
    
	  qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,jobfs=$JobFS,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v N_MshPrt,WorkDir,ScratchPath,dN,NCPU DynCohesive_DataPrepMPI.pbs; 
    
    sleep 2;

done
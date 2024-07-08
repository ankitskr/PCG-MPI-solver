export ModelName=OctBuild400
export NCPU=12 #1 (for serial); or 12, 24, 48, 96, 192 (for parallel)
export dN=1.0 #1.0, 0.5 or 0.25 (for parallel only: dN*min(NCPU,48) must be multiple of 4 -- A lower value of dN reduces memory requirement but delays parallelization)
export TotMem=1500GB
export JobFS=400GB
export Queue=hugemem
export WallTime=2:00:00


export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver_Contact/
cd ${WorkDir}PythonMPI/
N_MshPrts=(    1    2    3    6   12   24   48   96  192  384   768   1536   3072   6144  12288) # 20736
#		       0    1    2    3    4    5    6    7    8    9    10     11     12     13     14


for P0 in 4; do 
	
	  export N_MshPrt=${N_MshPrts[$P0]}
    export JobName=Msh_${ModelName}_N$N_MshPrt
    
	  qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,jobfs=$JobFS,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v N_MshPrt,WorkDir,ScratchPath,dN,NCPU Newmark_DataPrepMPI.pbs; 
    
    sleep 2;

done
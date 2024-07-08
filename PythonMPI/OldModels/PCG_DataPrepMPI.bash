export ModelName=PureBending_Octree_0
export NCPU=4
export TotMem=120GB
export Queue=hugemem
export WallTime=1:00:00


export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver/
cd ${WorkDir}PythonMPI/


for P0 in 3 4 5; do 
	
	  export P0=$P0
    export JobName=Msh_${ModelName}_N$P0
    
	  qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -v P0,WorkDir,ScratchPath PCG_DataPrepMPI.pbs; 
    
    sleep 2;

done
export ModelName=SpeedUpTest
NCPUS=(      48  48  48  48  48  48  96 144 288  528 1056 2064  4128)
TotMemList=(190 190 190 190 190 190 380 560 980 1800 3600 7200 14400)
#		          0   1   2   3   4   5   6   7   8    9   10   11    12
#export Queue=hugemem
export Queue=normal
export WallTime=4:00:00

export ProjName=ud04
export Email=ankit@unsw.edu.au
export ScratchPath=/g/data/ud04/Ankit/${ModelName}/
export LogPath=${ScratchPath}Log/
export WorkDir=~/MatlabLink_4Jul/ExplicitSolver/
cd ${WorkDir}PythonMPI/

export MeshSizeFacList=(12 13 14 15 16 17 18 19 20 21 22 23 24)

for r0 in 0 1 2 3; do

  export Run=$r0
  
  for Sz in 0 1 2 3 4 5 6 7 8 9 10 11 12; do
    
    export MeshSizeFac=${MeshSizeFacList[$Sz]}
    
    for P0 in 0 1 2 3 4; do 
    	
    	  export P0=$P0
        export JobName=${ModelName}_N$P0
    	  export NCPU=${NCPUS[$P0]}
        export TotMem=${TotMemList[$P0]}GB
    	  
        qsub -N $JobName -P $ProjName -q $Queue -l ncpus=$NCPU,walltime=$WallTime,mem=$TotMem,storage=gdata/$ProjName -M $Email -m ae -o $LogPath -j oe -V SpeedUpTest.pbs;
    	
        sleep 2;
  
    done
    
  done
  
  sleep 30;
  
done
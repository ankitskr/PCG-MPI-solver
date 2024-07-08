close
clear
clc

%% control parameters
ctrl.example = 'CubeTest';
ctrl.scratchpath  = [replace(pwd, '\','/') '/'];
ctrl.type = 'Oct';
% addpath(genpath(ctrl.scratchpath))

addpath(genpath([ctrl.scratchpath,'part/']));


ctrl.PartMesh           = 0; % 1 = run Mesh Patitioning
ctrl.MPIDataPrep        = 2; % 1 = Apply DataPrep Settings, 2 = Run in interactive mode
ctrl.Analysis           = 2; % 1 = Apply Analysis Settings, 2 = Run in interactive mode
ctrl.PostProcess        = 1; % 1 = Perform post-processing

ctrl.Run                = 1;

disp(ctrl.example)

ModelDataPath = [ctrl.scratchpath 'ModelData/'];
InputMeshPath = [ModelDataPath 'InputMesh/'];
MatDataPath = [ModelDataPath 'Mat/'];
    

%% Mesh Partition------------------------------------------
if ctrl.PartMesh == 1

    TIC_MeshPart = tic;

    if ~exist(MatDataPath, 'dir')
        error('MatDataPath not found!')
    end

    if ~exist('MeshData','var')        
        polys = ImgMeshImport(InputMeshPath, 'polys');
        sctrs = ImgMeshImport(InputMeshPath, 'sctrs');
    else
        polys = MeshData.polys;
        sctrs = MeshData.sctrs;
    end

    Dep = 3; % number of partition = 2^Dep
    NMeshParts = 2^Dep;
    Part = partition_mesh(polys, sctrs, Dep);
    if Dep==0
        binsave([MatDataPath  'MeshPart_' num2str(NMeshParts)], zeros(1,length(polys)), 'int32');
    else
        binsave([MatDataPath  'MeshPart_' num2str(NMeshParts)], Part{Dep}-1, 'int32');
    end
    NPrt = int16(NMeshParts); %No. of mesh-parts or cores 
    save([MatDataPath 'TempRunParam'], 'NPrt');

    ConfigGlobCMD = sprintf('python "PythonMPI/config_GlobData_win.py" "%s"', ctrl.scratchpath);
    system(ConfigGlobCMD);

    disp(['Time spent on Mesh Paritioning is: ' num2str(toc(TIC_MeshPart))]);


end


%% MPI Data Prep-------------------------------------------

if ~exist(MatDataPath, 'dir')
    error('MatDataPath not found!')
end

binsave([MatDataPath  'RefPlotDofVec'], [], 'int64');
binsave([MatDataPath  'qpoint'], [], 'float64');

NSplits = int16(4); %Multiple of 4 if running on Gadi (No. of NUMA nodes per Compute-nodes is 4)
NPrt = 8; %No. of partitions for parallel simulation

% MPI Data preparation
disp('Preparing MPI Data..')
%Creating MPI Data
%         DataPrepCMD = sprintf('"%s" -np %u --map-by numa python3 "PythonMPI/Elast_ParDataPrepMPI.py" %u "%s" %u', MPIExecPath, uint32(NSplits), uint32(NPrt), ctrl.scratchpath, 0);
DataPrepCMD = sprintf('mpiexec -np %u python3 "Elast_ParDataPrepMPI_win.py" %u "%s" %u', uint32(NSplits), uint32(NPrt), ctrl.scratchpath, 0);
DataPrepCMD
system(DataPrepCMD);




%% Analysis------------------------------------------------
if ismember(ctrl.Analysis, [1,2])

    if ~exist(MatDataPath, 'dir')
        error('MatDataPath not found!')
    end
    
    Export = 1; % export deformation in VTK
    ExportFrmRate = 1;
    ExportFrms = []; 
    Plot = 0;
    FintCalcMode = 'outbin'; %inbin, outbin or infor (Calculation mode for F_int)
    ExportVars = 'U'; %D U PS PE GS GE

    %time history
    
    %PCG
    Tol = 1e-7;
    MaxIter = 10000;  %numel(nodes);
    TimeStepDelta = [0:1]; %Starts from 0
    RefMaxTimeStepCount = length(TimeStepDelta);
    
    PlotFlag = logical(Plot); ExportFlag = logical(Export);
    delete([MatDataPath 'GlobSettings.mat'])
    save([MatDataPath 'GlobSettings'], 'Tol', 'MaxIter', 'TimeStepDelta', 'RefMaxTimeStepCount', 'ExportFrms', 'ExportFrmRate', 'PlotFlag', 'ExportFlag', 'FintCalcMode', 'ExportVars');

    if ctrl.Analysis == 2
        
        SpeedTest = 0;
        load([MatDataPath 'TempRunParam'], 'NPrt');

        %Running MPI
        TIC_MPIRun = tic;
        fprintf('Running MPI with %u processes..\n', uint32(NPrt))
        
%         MPICMD = sprintf('"%s" -np %u --map-by numa python3 "PythonMPI/MGStatic_Analysis.py" %s "%s" %u %u %s', MPIExecPath, uint32(NPrt), ctrl.example, ctrl.scratchpath, ctrl.Run, SpeedTest, '0');
        MPICMD = sprintf('mpiexec -np %u python "Elast_StaticAnalysis.py" %s "%s" %u %u %s', uint32(NPrt), ctrl.example, ctrl.scratchpath, ctrl.Run, SpeedTest, '0');
        MPICMD
        system(MPICMD);
        disp(['Time spent on MPI Run is: ' num2str(toc(TIC_MPIRun))]);
    end
end

%% Post processing-----------------------------------------
if ctrl.PostProcess == 1
    
    if ~exist(MatDataPath, 'dir')
        error('MatDataPath not found!')
    end

    ResVecDataPath = [ctrl.scratchpath 'Results_Run' num2str(ctrl.Run) '/ResVecData/'];
    if ~exist(ResVecDataPath, 'dir')
        error('ResVecDataPath not found!')
    end
    
    N_cores = 1;
    load([MatDataPath 'GlobSettings'], 'ExportVars');
    Mode = 'Full'; %Full or MidSlices or Boundary

    %Exporting VTKs
    TIC_VTKRun = tic;
    fprintf('Exporting VTKs/Mat with %u processes..\n', uint32(N_cores))
    
%     VTKCMD = sprintf('"%s" -np %u --map-by numa python3 "PythonMPI/ExportVTK.py" %s "%s" %u "%s" "%s"', MPIExecPath, uint32(N_cores), ctrl.example, ctrl.scratchpath, ctrl.Run, ExportVars, Mode);
    VTKCMD = sprintf('mpiexec -np %u python "ExportVTK.py" %s "%s" %u "%s" "%s"', uint32(N_cores), ctrl.example, ctrl.scratchpath, ctrl.Run, ExportVars, Mode);
    VTKCMD
    system(VTKCMD);
%     MatCMD = sprintf('"%s" -np %u --map-by numa python3 "PythonMPI/ExportMat.py" %s "%s" %u "%s"', MPIExecPath, uint32(N_cores), ctrl.example, ctrl.scratchpath, ctrl.Run, ExportVars);
    MatCMD = sprintf('mpiexec -np %u python "ExportMat.py" %s "%s" %u "%s"', uint32(N_cores), ctrl.example, ctrl.scratchpath, ctrl.Run, ExportVars);
    MatCMD
    system(MatCMD);
    disp(['Time spent on exporting Files is: ' num2str(toc(TIC_VTKRun))]);


end

function [RefPlotDofVec, qpoint] = manual_exp(MatDataPath, nodes, L)
    
    % points to plot
    qpoint = [];
    RefPlotDofVec = [];
    
end

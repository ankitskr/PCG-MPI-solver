# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:06:16 2020

@author: z5166762
"""


import numpy as np
import pandas as pd
import glob

ModelName           = 'DmgTest'
#ScratchFolder       = '/g/data/ud04/Ankit/' + ModelName + '/'
ScratchFolder       = '/home/561/aa5206/MatlabLink_4Jul/ExplicitSolver_Contact/debug/' + ModelName + '/'
N_MeshPrtList       = [12, 24, 48, 96, 192]
#P0List              = [6, 7, 9]
FintCalcModeList    = ['infor', 'inbin', 'outbin']
FintCalcMode        = FintCalcModeList[2]
N_MshPrtListCount   = len(N_MeshPrtList)
Run                 = int(6)

Table = {}
Table['N_Cores'] = ['']*N_MshPrtListCount
Table['Init'] = ['']*N_MshPrtListCount
Table['Calc_Mean'] = ['']*N_MshPrtListCount
Table['CommWait_Mean'] = ['']*N_MshPrtListCount
Table['Elast_Mean'] = ['']*N_MshPrtListCount
Table['Dmg_Mean'] = ['']*N_MshPrtListCount
#Table['CommWait_Min'] = ['']*N_MshPrtListCount
#Table['CommWait_Max'] = ['']*N_MshPrtListCount
Table['TotalCPU'] = ['']*N_MshPrtListCount
Table['Memory'] = ['']*N_MshPrtListCount
Table['PBS_JobId'] = ['']*N_MshPrtListCount

#Table['Wall'] = []
#Table['RelRes'] = []
#Table['Iter'] = []


#ResultFolder = ScratchFolder + 'Results_Run' + str(Run) + '_SpeedTest/'
ResultFolder = ScratchFolder + 'Results_Run' + str(Run) + '/'
for p in range(N_MshPrtListCount):
    
    N_MeshPrt = N_MeshPrtList[p]
    #FileName = ResultFolder + 'DispVecData/' +  ModelName + '_MP' + str(N_MeshPrt) + '_' + FintCalcMode + '_TimeData.npz'
    FileName = ResultFolder + 'PlotData/' +  ModelName + '_MP' + str(N_MeshPrt) + '_' + FintCalcMode + '_TimeData.npz'
    #FileName = ScratchFolder + 'Results/' + ModelName + '_MP' + str(N_MeshPrt) + '_' + FintCalcMode + '.npz'
    TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
    Table['N_Cores'][p]         = N_MeshPrt 
    Table['Init'][p]            = TimeData['Mean_FileReadTime']
    Table['Calc_Mean'][p]       = TimeData['Mean_CalcTime']
    Table['CommWait_Mean'][p]   = TimeData['Mean_CommWaitTime']
    Table['Elast_Mean'][p]      = TimeData['Mean_ElastTime']
    Table['Dmg_Mean'][p]        = TimeData['Mean_DmgTime']
    #Table['CommWait_Min'][p]    = TimeData['MinCommWaitTime']
    #Table['CommWait_Max'][p]    = TimeData['MaxCommWaitTime']
    Table['TotalCPU'][p]        = TimeData['TotalTime']
    Table['PBS_JobId'][p]       = TimeData['PBS_JobId']
    #Table['Wall'].append(0.0)
    #Table['RelRes'].append(TimeData['RelRes'])
    #Table['Iter'].append(TimeData['Iter'])
    

"""
#Reading Memory
LogFolder = ScratchFolder + 'Log/'
LogFileList = [f for f in glob.glob(LogFolder+'*.OU')]
for p in range(N_MshPrtListCount):
    for LogFile in LogFileList:
        if Table['PBS_JobId'][p] in LogFile:
            break
    else:   raise Exception
    
    f = open(LogFile, "r")
    LineList = f.readlines()
    for Line in LineList:
        if 'Memory Used' in Line:
            if 'GB' in Line:    MemUnit = 'GB'
            elif 'TB' in Line:    MemUnit = 'TB'
            Mem = float(Line.split(':')[-1].split(MemUnit)[0])
            Table['Memory'][p] = Mem
            break
"""            


df = pd.DataFrame(Table, columns = Table.keys())

OutputFile = ResultFolder + 'CPUTimeData_' + FintCalcMode + '.xlsx'
df.to_excel (OutputFile, index = False, header=True)
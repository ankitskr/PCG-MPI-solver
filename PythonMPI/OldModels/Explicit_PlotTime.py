# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:06:16 2020

@author: z5166762
"""


import numpy as np
import pandas as pd

ModelName = 'Everest3200'
FintCalcModeList = ['infor', 'inbin', 'outbin']
P0_Max = 10


FintCalcMode = FintCalcModeList[1]

Table = {}
Table['N_Cores'] = []
Table['Init'] = []
Table['Calc_Mean'] = []
Table['CommWait_Mean'] = []
Table['TotalCPU'] = []
Table['Max_TotalTime_i'] = []
Table['TimeSteps'] = []

for p in range(P0_Max+1):

    
    FileName = ModelName +'_Explicit/'+ ModelName + '_MP' + str(p) + '_' + FintCalcMode + '.npz'
    TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
    
    Table['TimeSteps'].append(1000)
    
    
    Table['N_Cores'].append(2**p)
    Table['Init'].append(TimeData['Mean_FileReadTime'])
    Table['Calc_Mean'].append(TimeData['Mean_CalcTime'])
    Table['CommWait_Mean'].append(TimeData['Mean_CommWaitTime'])
    Table['TotalCPU'].append(TimeData['TotalTime'])
    Table['Max_TotalTime_i'].append(TimeData['Max_TotalTime_i'])
    
    
    


df = pd.DataFrame(Table, columns = Table.keys())

df.to_excel (r'df_timedata.xlsx', index = False, header=True)
    
    
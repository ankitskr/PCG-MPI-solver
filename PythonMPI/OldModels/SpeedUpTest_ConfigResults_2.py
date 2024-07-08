# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:06:16 2020

@author: z5166762
"""

import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(20,6))

ax0 = plt.subplot(131)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)
 

ScratchPath = '/g/data/ud04/Ankit/SpeedUpTest/'
ModelName = 'SpeedUpTest'
P0_Max = 10
N_CoreList = [2**p for p in range(P0_Max+1)]
Total_NDOF_List = []
SizeList = (14, 15, 16)


for Sz in SizeList:
    
    Mean_CalcTimeList = []
    
    for p in range(P0_Max+1):
        
        Temp_Mean_CalcTimeList = []
        for r in range(8):
            
            FileName = ScratchPath + 'Results_2/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
            TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
            
            Temp_Mean_CalcTimeList.append(TimeData['Mean_CalcTime'])
        
        Mean_CalcTimeList.append(np.min(Temp_Mean_CalcTimeList))
        
    
    SpeedUpList = Mean_CalcTimeList[0]/np.array(Mean_CalcTimeList)
    StrEffList = SpeedUpList/np.array(N_CoreList)
    
    Total_NDOF_Apprx = np.round((1e-6)*TimeData['Total_NDOF'], 2)
    Total_NDOF_List.append(Total_NDOF_Apprx)
    
    #ax0.plot(np.log2(N_CoreList), np.log2(Mean_CalcTimeList), label = str(Total_NDOF_Apprx))
    #ax0.plot(np.log2(N_CoreList), np.log2(SpeedUpList), label = str(Total_NDOF_Apprx))
    ax0.plot(np.log2(N_CoreList), StrEffList, label = str(Total_NDOF_Apprx))

 
ax0.legend(title='NDOF (1e6)')

plt.xlabel('N_Cores (Log2)')
plt.ylabel('Srong Efficiency')


ax0 = plt.subplot(132)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)

for Sz in SizeList:
    
    VarSizeList = []
    
    for p in range(P0_Max+1):
        
        r = 7
    
        FileName = ScratchPath + 'Results_2/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
        TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
        VarSizeData = np.load(FileName, allow_pickle=True)['VarSizeData'].item()
        
        VarSize = VarSizeData['Ke'] + VarSizeData['ElemDofVectorList'] + VarSizeData['MP_U'] + VarSizeData['ElemList_U'] + VarSizeData['ElemList_Fint']
        VarSizeList.append(VarSize/(1e6))
    
    Total_NDOF_Apprx = np.round((1e-6)*TimeData['Total_NDOF'], 2)
    
    ax0.plot(np.log2(N_CoreList), np.log2(VarSizeList), label = str(Total_NDOF_Apprx))

#NCPUS=(         8   8   8   8  16  48   96  144  288  528  1056
CacheMemList = [35, 35, 35, 35, 35, 70, 105, 210, 385, 770, 1505]
plt.plot(np.log2(N_CoreList), np.log2(CacheMemList), label='CacheMem')
ax0.legend(title='NDOF (1e6)')

plt.xlabel('N_Cores (Log2)')
plt.ylabel('Size (MB) (Log2)')


ax0 = plt.subplot(133)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)

NSize = len(SizeList)
Mean_CalcTimeList = [] 
N_CoreList = []
for i in range(NSize):

    p = i
    Sz = SizeList[i]
    N_CoreList.append(2**p)
    
    Temp_Mean_CalcTimeList = []
    for r in range(8):
        FileName = ScratchPath + 'Results_2/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
        TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()        
        Temp_Mean_CalcTimeList.append(TimeData['Mean_CalcTime'])
    
    Mean_CalcTimeList.append(np.min(Temp_Mean_CalcTimeList))
    
WeakEffList = Mean_CalcTimeList[0]/np.array(Mean_CalcTimeList)
print(Mean_CalcTimeList)
ax0.plot(np.log2(N_CoreList), WeakEffList)

plt.xlabel('N_Cores (Log2)')
plt.ylabel('Weak Efficiency')

fig.savefig(ScratchPath+'Results_2.png', dpi = 360, bbox_inches='tight')




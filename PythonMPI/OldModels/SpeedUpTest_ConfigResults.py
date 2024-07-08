# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 23:06:16 2020

@author: z5166762
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os

fig = plt.figure(figsize=(25,6))

ax0 = plt.subplot(131)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)
 

ScratchPath = '/g/data/ud04/Ankit/SpeedUpTest/'
ModelName = 'SpeedUpTest'
Total_NDOF_List = []
#SizeList = (12, 13, 14, 15, 16, 17, 18, 19, 20, 21)
SizeList = range(12,25, 2)
RunList = range(3)
P0_Max = 10

N_CoreList = [2**p for p in range(P0_Max+1)]

#Verifying Files
for Sz in SizeList:
    
    Mean_CalcTimeList = []
    
    for p in range(P0_Max+1):
        
        Temp_Mean_CalcTimeList = []
        for r in RunList:
            
            FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
            if not os.path.exists(FileName):
                print(FileName)
            

for Sz in SizeList:
    
    Mean_CalcTimeList = []
    
    for p in range(P0_Max+1):
        
        Temp_Mean_CalcTimeList = []
        for r in RunList:
            
            FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
            TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
            
            Temp_Mean_CalcTimeList.append(TimeData['Mean_CalcTime'])
        
        Mean_CalcTimeList.append(np.min(Temp_Mean_CalcTimeList))
        
    
    SpeedUpList = Mean_CalcTimeList[0]/np.array(Mean_CalcTimeList)
    StrEffList = SpeedUpList/np.array(N_CoreList)
    
    Total_NDOF_Apprx = np.round((1e-6)*TimeData['Total_NDOF'], 2)
    Total_NDOF_List.append(Total_NDOF_Apprx)
    
    #ax0.plot(np.log2(N_CoreList), np.log2(Mean_CalcTimeList), label = str(Total_NDOF_Apprx))
    #ax0.plot(np.log2(N_CoreList), np.log2(SpeedUpList), label = str(Total_NDOF_Apprx))
    ax0.plot(N_CoreList, StrEffList, label = str(Total_NDOF_Apprx))

 
ax0.legend(title='Total_NDOF (1e6)')

ax0.set_xscale('log', basex=2)
ax0.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(x)))


plt.xlabel('N_Cores')
plt.ylabel('Efficiency (Strong Scaling)')
plt.title('(a)')


#--------------------------------------------------------------------------------------------


ax0 = plt.subplot(132)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)
N_CoreList = [2**p for p in range(P0_Max+1)]

for Sz in SizeList:    
    VarSizeList = []
    
    for p in range(P0_Max+1):        
        r = 0    
        FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p) + '_Sz' + str(Sz) + '_R' +str(r)+ '.npz'
        TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
        VarSizeData = np.load(FileName, allow_pickle=True)['VarSizeData'].item()
        
        VarSize = sum([VarSizeData[Key] for Key in VarSizeData.keys()])/1e6
        VarSizeList.append(VarSize)
    
    Total_NDOF_Apprx = np.round((1e-6)*TimeData['Total_NDOF'], 2)
    
    ax0.plot(N_CoreList, VarSizeList, label = str(Total_NDOF_Apprx))


#NCPUS=(                      1    2    4   8  16  32   64  128  256  512  1024
L2_CacheMemList = np.array([  1,   2,  4,   8, 16, 32,  64, 128, 256, 512, 1024])
L3_CacheMemList = np.array([ 36,  36,  36, 36, 36, 72, 143, 215, 429, 787, 1573])

CacheMemList = (L2_CacheMemList + L3_CacheMemList)*0.8

N_CoreList = [2**p for p in range(len(CacheMemList))]
plt.plot(N_CoreList, CacheMemList, '-*k', label='Cache Mem')

SizeList = range(12,25)
VarSizeList_WeakEff0 = []
N_CoreList_WeakEff0 = []
VarSizeList_WeakEff1 = []
N_CoreList_WeakEff1 = []
p0 = -1
p1 = -1
NSize = len(SizeList)
for i in range(NSize):
    r = 0
    if i <= 10:
        
        p0 += 1
        Sz0 = SizeList[i]
        FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p0) + '_Sz' + str(Sz0) + '_R' +str(r)+ '.npz'
        TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
        VarSizeData = np.load(FileName, allow_pickle=True)['VarSizeData'].item()

        VarSize = sum([VarSizeData[Key] for Key in VarSizeData.keys()])/1e6
        VarSizeList_WeakEff0.append(VarSize)
        N_CoreList_WeakEff0.append(2**p0)
        
    if i >= 6:
        
        p1 += 1
        Sz1 = SizeList[i]
        FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p1) + '_Sz' + str(Sz1) + '_R' +str(r)+ '.npz'
        TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()
        VarSizeData = np.load(FileName, allow_pickle=True)['VarSizeData'].item()

        VarSize = sum([VarSizeData[Key] for Key in VarSizeData.keys()])/1e6
        VarSizeList_WeakEff1.append(VarSize)
        N_CoreList_WeakEff1.append(2**p1)
        
        
    
    
    
ax0.plot(N_CoreList_WeakEff0, VarSizeList_WeakEff0, '-^b', label = 'RefData-0')
ax0.plot(N_CoreList_WeakEff1, VarSizeList_WeakEff1, '-^r', label = 'RefData-1')


ax0.legend(title='Total_NDOF (1e6)')
ax0.set_yscale('log', basey=2)
ax0.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: '{:2.0f}'.format(y)))
ax0.set_xscale('log', basex=2)
ax0.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(x)))
plt.ylim(2, 14000)
plt.xlabel('N_Cores')
plt.ylabel('Total_VarSize (MB)')
plt.title('(b)')



#--------------------------------------------------------------------------------------------

ax0 = plt.subplot(133)
ax0.yaxis.grid(linestyle='--')
ax0.xaxis.grid(linestyle='--')
ax0.set_axisbelow(True)

SizeList = range(12,25)
NSize = len(SizeList)
Mean_CalcTimeList0 = [] 
N_CoreList0 = []
Mean_CalcTimeList1 = [] 
N_CoreList1 = []
p0 = -1
p1 = -1

for i in range(NSize):

    if i <= 10:
            
        p0 += 1
        Sz0 = SizeList[i]
        N_CoreList0.append(2**p0)
        
        Temp_Mean_CalcTimeList = []
        for r in RunList:
            FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p0) + '_Sz' + str(Sz0) + '_R' +str(r)+ '.npz'
            TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()        
            Temp_Mean_CalcTimeList.append(TimeData['Mean_CalcTime'])
        
        Mean_CalcTimeList0.append(np.min(Temp_Mean_CalcTimeList))
        
    if i >= 6:
            
        p1 += 1
        Sz1 = SizeList[i]
        N_CoreList1.append(2**p1)
        
        Temp_Mean_CalcTimeList = []
        for r in RunList:
            FileName = ScratchPath + 'Results/' + ModelName + '_MP' + str(p1) + '_Sz' + str(Sz1) + '_R' +str(r)+ '.npz'
            TimeData = np.load(FileName, allow_pickle=True)['TimeData'].item()        
            Temp_Mean_CalcTimeList.append(TimeData['Mean_CalcTime'])
        
        Mean_CalcTimeList1.append(np.min(Temp_Mean_CalcTimeList))
        
    

WeakEffList0 = Mean_CalcTimeList0[0]/np.array(Mean_CalcTimeList0)
ax0.plot(N_CoreList0, WeakEffList0,'-^b', label = 'RefData-0')

WeakEffList1 = Mean_CalcTimeList1[0]/np.array(Mean_CalcTimeList1)
ax0.plot(N_CoreList1, WeakEffList1,'-^r', label = 'RefData-1')

ax0_1 = ax0.twinx()
N_CoreList = [2**p for p in range(len(CacheMemList))]
CacheMemPerCore = CacheMemList/N_CoreList
ax0_1.plot(N_CoreList, CacheMemPerCore,'--k', label = 'CacheMem per Core')


ax0.set_xscale('log', basex=2)
ax0.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: '{:2.0f}'.format(x)))

plt.xlabel('N_Cores')
plt.ylabel('Efficiency (Weak Scaling)')
plt.legend()

fig.savefig(ScratchPath+'Results_3.png', dpi = 360, bbox_inches='tight')




# -*- coding: utf-8 -*-
import sys

MyHostFile =      sys.argv[1]
NProc =           int(sys.argv[2])

with open(MyHostFile, 'r') as file:
    HostList = file.readlines()
    NHosts = len(HostList)
    NProc_PerHost = int(NProc/NHosts)
    Rem = NProc%NHosts
    SlotList = [NProc_PerHost]*NHosts
    for i in range(Rem): SlotList[i] += 1

file.close()

with open(MyHostFile, 'w') as file:
    for i in range(NHosts):    
        HostList[i] = HostList[i][:-1] + ' slots=' + str(SlotList[i]) + '\n'
    file.writelines(HostList)

file.close()


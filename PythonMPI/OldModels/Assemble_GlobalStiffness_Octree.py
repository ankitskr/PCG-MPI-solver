

"""
def assembleSubDomainMatrix(DofGlb, Type, Level, Sign, Ke, M, GlobData):
    
    GlobNDOF = GlobData['GlobNDOF']
    GlobK = np.zeros([GlobNDOF, GlobNDOF])
    
    N_Elem = len(DofGlb)
    UniqueTypeList = np.unique(Type)
    N_Type = len(UniqueTypeList)
    
    for j in range(N_Type):
        
        RefTypeId = UniqueTypeList[j]
        I = np.where(Type==RefTypeId)[0]
        
        SignVectorList = np.array(tuple(Sign[I]), dtype=bool).T
        LevelList = np.array(tuple(Level[I]), dtype=float)
        DofGlbList = np.array(tuple(DofGlb[I]), dtype=float)
        
        for i in range(N_Elem):
            
            ElemDof = DofGlb[i]
            N_ElemDofs = len(ElemDof)
            RefKe = Ke[RefTypeId]*LevelList[j] ???
        
            GlobK[ElemDof, ElemDof] += RefKe

    
    InvM = np.diag(np.array(1.0/M, dtype=float))
    A = np.dot(InvM, GlobK)
"""
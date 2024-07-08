clear
clc
scratchpath  = '/g/data/ud04/Ankit/Contact_cuboid_Hex_6/';
MatDataPath = [scratchpath 'ModelData/Mat/'];
load([MatDataPath 'MeshPart_12.mat'], 'RefPart')
N_Parts = length(unique(RefPart));

IntfcElem = MatDataImport(MatDataPath, 'IntfcElem');
N_IntfcElem = length(IntfcElem);
IntfcPartIdData = zeros(1,N_Parts);
for i=1:N_IntfcElem
    IntfcElemData = IntfcElem{i};
    if ~isempty(IntfcElemData)
        Intfc_PartId = RefPart(IntfcElemData{12})+1;
        IntfcPartIdData(Intfc_PartId) = IntfcPartIdData(Intfc_PartId) + 1;
    end
end

ElemPartIdData = histcounts(RefPart);

(IntfcPartIdData*335+ElemPartIdData)'/1e6
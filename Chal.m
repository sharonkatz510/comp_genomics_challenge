%%
data = readtable("Known_set_Bacillus.xlsx");
index = cellfun(@str2num,data.GeneIndex);
PA = cellfun(@str2num,data.GeneIndex);



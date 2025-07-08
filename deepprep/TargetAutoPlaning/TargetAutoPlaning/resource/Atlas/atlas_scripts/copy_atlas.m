inpath = '/autofs/space/bidlin9_002/users/Jianxun/IndiAtlas/Results/R1_Atlas/Atlas_LR';
outpath = '.';
lobes = {'Frontal','Motor','Parietal','Temporal','Visual'};
%nums = [13 6 11 10 6];
nums = [17 12 15 20 12];
hemis = {'lh', 'rh'};

for i = 2:5
    lobe = lobes{i}
    num = nums(i)
    snum = num2str(num);
    for h = 1:2
        hemi = hemis{h}
        infile = [inpath '/' lobe '/fs6_by_fs3/Cluster' snum '/' hemi '.Clustering_' snum '_fs6.mgh'];
        outfile = [outpath '/' lobe '_' hemi '/fs6_by_fs3/Cluster' snum '/'];
        mkdir([outpath '/' lobe '_' hemi '/fs6_by_fs3/Cluster' snum])
        eval(['!cp ' infile ' ' outfile])
        subpath = [outpath '/' lobe '_' hemi '/fs6_by_fs3']; 
        SplitClusters(subpath, num, hemi);
    end
end
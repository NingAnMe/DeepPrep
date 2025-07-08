inpath = '.';
outpath = '.';
lobes = {'Visual', 'Motor', 'Frontal', 'Temporal', 'Parietal' };
nums = [6,6,13,10,11];
%nums = [12,12,17,20,15];
hemis = {'lh', 'rh'};

cumsums = [0, cumsum(nums)];

for h = 1:2
    hemi = hemis{h}
    clusters = zeros(40962,1);
    for i = 1:5
        lobe = lobes{i}
        num = nums(i)
        snum = num2str(num);
        infile = [inpath '/' lobe '_' hemi '/fs6_by_fs3/Cluster' snum];
        tmp = load_mgh([infile  '/' hemi '.Clustering_' snum '_fs6.mgh']);
        clusters(tmp > 0) = tmp(tmp>0) + cumsums(i);
    end
    realpath = [outpath '/WB_' hemi '/fs6_by_fs3/Cluster' num2str(sum(nums))];
    mkdir(realpath)
    save_mgh(clusters, [realpath '/' hemi '.Clustering_' num2str(sum(nums)) '_fs6.mgh'], eye(4))
end
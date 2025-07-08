Areas = {'Visual', 'Motor', 'Parietal', 'Temporal', 'Frontal'};
Area_Num = [16, 21, 24, 30, 17];
hemi = 'rh';
Path = '.';
WB = zeros(40962,1);

for A = 1:length(Areas)
    Area = Areas{A}
    Area_Nums = num2str(Area_Num(A));
    Inpath = [Path '/' Area '_' hemi '/fs6_by_fs3/Cluster' Area_Nums];
    AreaParc = load_mgh([Inpath '/' hemi '.Clustering_' Area_Nums '_fs6.mgh']);
    if A == 1
        WB(AreaParc ~= 0) = AreaParc(AreaParc ~= 0);
    else
        WB(AreaParc ~= 0) = AreaParc(AreaParc ~= 0) + sum(Area_Num(1:A-1));
    end 
end
%Area_Num = Area_Num;
outpath = [Path '/WB_' hemi '/fs6_by_fs3/Cluster' num2str(sum(Area_Num))];
mkdir(outpath)
save_mgh(WB, [outpath '/' hemi '.Clustering_' num2str(sum(Area_Num)) '_fs6.mgh'], eye(4))
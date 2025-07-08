clear,clc;
Inpath = '.';

hemi = 'rh';
maskname = 'WB';
for n = [46]
    nCluster = num2str(n);
    subpath = [Inpath '/' maskname '_' hemi '/fs6_by_fs3/Cluster' nCluster ];
    func_WriteParcellation_to_annotation_atlas_fs6(subpath, n, hemi)
end

clear,clc

nums = '46';
hemi = 'lh';
[v, Lo,cto]= read_annotation(['~/Data/Share/Templates/' ...
    'Surf/fsaverage6/label/' hemi '.aparc.annot']);
ct = cto;
ct.table(1,:) = [10, 10, 10, 0, 657930];

Inpath = ['./WB_' hemi '/fs6_by_fs3/Cluster' nums];
mask = load_mgh([Inpath '/Network_All_boundary_' hemi '.mgh']);
mask(mask > 0) = 1;
L = 0*Lo;
L(mask == 1) = 657930;
write_annotation([Inpath '/Network_All_boundary_' hemi '_fs6.annot'],v,L,ct)
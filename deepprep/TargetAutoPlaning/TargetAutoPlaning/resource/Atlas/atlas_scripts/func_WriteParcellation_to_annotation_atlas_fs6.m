% should run Proj_to_individualBrain.sh before this
function[] = func_WriteParcellation_to_annotation_atlas_fs6(subpath, n, hm)

recon_dir = '~/Data/Share/tools/apps/arch/linux_x86_64/freesurfer/5.3.0/subjects';
recon_sub = 'fsaverage6';
ncluster = n;
load 113NetworkLUT_Manual

offset =0% 14;
confthresh =0.1
highconfthresh =0.6


dataPath = subpath;
outPath = dataPath;

[v, L,cto]= read_annotation([recon_dir,'/', recon_sub,'/label/',hm,'.aparc.annot']);
ct = newct;
ct.table(1+offset,4)=255;

L(1:end) = ct.table(1,5);
HL =L;

% Merge all the parcellation networks to one map

netmember = load_mgh([dataPath, '/' hm '.Clustering_' num2str(n) '_fs6.mgh']);

for k =1:ncluster+1

    idx= find(netmember==(k-1));
    L(idx) = ct.table(k+offset,5);

end

bounadry = load_mgh([dataPath, '/Network_All_boundary_' hm '.mgh']);
bounadry = reshape(bounadry, size(L));
netmember = reshape(netmember, size(L));
L(bounadry==1 | netmember == 0) = ct.table(113,5);
%L(bounadry==1) = ct.table(113,5);

fname =[outPath,  '/',hm,'_parc_result.annot']
write_annotation(fname, v, L, ct);
end
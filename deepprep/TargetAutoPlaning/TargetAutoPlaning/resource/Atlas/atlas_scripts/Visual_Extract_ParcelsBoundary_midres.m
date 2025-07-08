clear,clc

load ~/Jianxun/CoreMaterial/FirstadjacentVertex/fsaverage6/fs6_Firstadjacent_vertex.mat
hemi = 'rh';
nums = '46';
Inpath = ['./WB_' hemi '/fs6_by_fs3/Cluster' nums];

Clusters = load_mgh([Inpath '/' hemi '.Clustering_' nums '_fs6.mgh']);
GradMap = 0*Clusters;
for ind = 1:length(Clusters)
    neighbor = fs6_Firstadjacent_vertex_lh(:,ind);
    neighbors = neighbor(neighbor ~= 0);
    Grad = Clusters(ind) - Clusters(neighbors);
    if sum(Grad>0) > 0
        GradMap(ind) = 1;
    end
end

save_mgh(GradMap, [Inpath '/Network_All_boundary_' hemi '.mgh'], eye(4))

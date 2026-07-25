% Ground-truth run of the MATLAB meshing chain on a synthetic ball, so the
% Python port in analysis/volumetric/mesh.py can be checked against it.
% Writes JSON-ish text to stdout only; no files land in the TCell repo.

addpath(genpath('C:\Users\Upadhyaya_Lab\Code\TCell-3D-Morphodynamics\src'));

R = 14; N = 40; c = (N-1)/2;
[i1, i2, i3] = ndgrid(0:N-1, 0:N-1, 0:N-1);
ball = ((i1-c).^2 + (i2-c).^2 + (i3-c).^2) <= R^2;

fprintf('MASK voxels %d\n', nnz(ball));

% --- intermediates, mirroring generate_mesh.m step by step -----------------
[~, V0, F0] = evalc('v2s(ball, .99, 5)');
fprintf('V2S nodes %d faces %d cols %d\n', size(V0,1), size(F0,1), size(F0,2));

A_patch = patch_area(F0, V0);
keep_ratio = sum(A_patch > 0.2 * max(A_patch)) / numel(A_patch);
fprintf('KEEPRATIO_GIBBON %.12f\n', keep_ratio);

A_tri = patch_area(F0(:,1:3), V0);
kr_tri = sum(A_tri > 0.2 * max(A_tri)) / numel(A_tri);
fprintf('KEEPRATIO_TRIONLY %.12f\n', kr_tri);

[~, V1, F1] = evalc('meshresample(V0, F0, keep_ratio)');
fprintf('RESAMPLE nodes %d faces %d cols %d\n', size(V1,1), size(F1,1), size(F1,2));

% --- what the pipeline actually returns ------------------------------------
[V, F] = generate_mesh(ball, 5, 0.2, 10);
fprintf('FINAL nodes %d faces %d cols %d\n', size(V,1), size(F,1), size(F,2));

vol = triSurfVolume(F, V);
areas = patch_area(F, V);
SA = sum(areas);
sph = pi^(1/3) * (6 * vol)^(2/3) / SA;
fc = meshcentroid(V, F);
ext = max(fc, [], 1) - min(fc, [], 1);

fprintf('VOLUME %.9f\n', vol);
fprintf('SA %.9f\n', SA);
fprintf('SPHERICITY %.9f\n', sph);
fprintf('HEIGHT %.9f\n', ext(3));
fprintf('EXTENT %.9f %.9f %.9f\n', ext(1), ext(2), ext(3));
fprintf('VMIN %.9f %.9f %.9f\n', min(V,[],1));
fprintf('VMAX %.9f %.9f %.9f\n', max(V,[],1));
fprintf('HASHOLES %d\n', mesh_has_holes(V, F));
fprintf('ANALYTIC_VOL %.9f\n', 4/3*pi*R^3);
fprintf('ANALYTIC_SA %.9f\n', 4*pi*R^2);
fprintf('DONE\n');

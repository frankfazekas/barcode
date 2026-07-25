% Run the MATLAB curvature chain on meshes exported from the Python port, so the
% two curvature implementations are compared on identical input.

addpath(genpath('C:\Users\Upadhyaya_Lab\Code\TCell-3D-Morphodynamics\src'));
OUT = ['C:\Users\UPADHY~1\AppData\Local\Temp\claude\' ...
       'C--Users-Upadhyaya-Lab-Code-barcode\' ...
       '9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad'];

cases = {'11_2', '11_8', '1_1', '12_1'};

for k = 1:numel(cases)
    S = load(fullfile(OUT, ['mesh_cell' cases{k} '.mat']));
    V = double(S.V); F = double(S.F);

    face_areas = patch_area(F, V);
    face_centroids = meshcentroid(V, F);
    bt = identify_bottom_top_faces(face_centroids);

    [C_min_V, C_max_V, ~, ~] = compute_curvatures_Rusinkiewicz(V, F);
    [C_min_F, C_max_F, C_mean_F, ~] = curvatures_on_faces(F, C_min_V, C_max_V);
    C_class_F = classify_concavity(C_min_F, C_max_F);

    [m_mean, m_min, m_max] = mean_curvature_over_mesh(C_mean_F, C_min_F, C_max_F, ...
                                                      face_areas, bt);
    [invag, concave] = find_invag_ratio(C_class_F, face_areas, bt);

    fprintf('ML cell%s faces=%d mean=%.9f min=%.9f max=%.9f invag=%.9f concave=%.9f bt=%.9f\n', ...
            cases{k}, size(F,1), m_mean, m_min, m_max, invag, concave, ...
            sum(bt)/numel(bt));

    % Face-by-face agreement with the Python arrays carried in the same file.
    d = abs(C_mean_F - double(S.py_k_mean_F(:)));
    rel = d ./ max(abs(double(S.py_k_mean_F(:))), 1e-12);
    fprintf('ML cell%s perface_k_mean maxabs=%.3e median_abs=%.3e p99_rel=%.3e\n', ...
            cases{k}, max(d), median(d), prctile(rel, 99));
end
fprintf('DONE\n');

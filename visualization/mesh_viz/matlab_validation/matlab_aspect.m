% MATLAB's own compute_MIP_properties on the same masks, for comparison with the port.
addpath(genpath('C:\Users\Upadhyaya_Lab\Code\TCell-3D-Morphodynamics\src'));

base = ['F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\NaBu800 Experiments\' ...
        'Control_3D_CD3\all_cells_together\BARCODE\masks\'];
psize = 0.065;

fprintf('%8s %12s %12s %12s %12s\n', 'frame', 'MIP_area', 'major', 'minor', 'MIPcount_area');
for fr = [1 5 10 15]
    m = imread3(sprintf('%sCell1_%d_SegMask.tif', base, fr));
    % Our arrays are (Z,Y,X); MATLAB stacks read as (Y,X,Z), so max over dim 3 is the
    % same XY projection either way.
    [a, mj, mn] = compute_MIP_properties(logical(m), psize);
    proj = max(logical(m), [], 3);
    fprintf('%8d %12.6f %12.6f %12.6f %12.6f\n', fr, a, mj, mn, psize^2*nnz(proj));
end
fprintf('DONE\n');

function stack = imread3(path)
    info = imfinfo(path);
    n = numel(info);
    stack = false(info(1).Height, info(1).Width, n);
    for k = 1:n
        stack(:, :, k) = imread(path, k) > 0;
    end
end

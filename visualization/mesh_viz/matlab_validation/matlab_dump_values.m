f = ['F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA\' ...
     'Control\cells\all_cells_together\prog_live_cells\' ...
     'Jurkats_live_Control_04142022_results.mat'];
S = load(f);

cells  = [11 11 11 1 3 5 12];
frames = [ 1  2  8 1 1 1  1];

fprintf('%6s %6s %12s %12s %12s %12s %12s %12s\n', ...
        'cell', 'frame', 'vol_mesh', 'vol_seg', 'sphericity', 'SA_derived', 'height', 'sphere_rad');
for k = 1:numel(cells)
    c = cells(k); fr = frames(k);
    v   = S.nuc_volume_mesh(c, fr);
    vs  = S.nuc_volume_from_seg(c, fr);
    sp  = S.nuc_mesh_sphericity(c, fr);
    h   = S.nuc_height(c, fr);
    r   = S.nuc_same_vol_sphere_rad(c, fr);
    sa  = pi^(1/3) * (6*v)^(2/3) / sp;
    fprintf('%6d %6d %12.4f %12.4f %12.4f %12.4f %12.4f %12.4f\n', c, fr, v, vs, sp, sa, h, r);
end
fprintf('DONE\n');

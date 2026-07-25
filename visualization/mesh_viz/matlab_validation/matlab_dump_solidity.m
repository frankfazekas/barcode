% Dump the MATLAB pipeline's per-(cell, frame) nuc_solidity for comparison.
f = ['F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA\' ...
     'Control\cells\all_cells_together\prog_live_cells\' ...
     'Jurkats_live_Control_04142022_results.mat'];
out = ['C:\Users\UPADHY~1\AppData\Local\Temp\claude\' ...
       'C--Users-Upadhyaya-Lab-Code-barcode\' ...
       '9f48b303-899e-4482-ab3b-afb87486e1b4\scratchpad\matlab_solidity.csv'];
S = load(f);
[nc, nf] = size(S.nuc_solidity);
fid = fopen(out, 'w');
fprintf(fid, 'cell,frame,ml_solidity\n');
for c = 1:nc
    for fr = 1:nf
        fprintf(fid, '%d,%d,%.12g\n', c, fr, S.nuc_solidity(c, fr));
    end
end
fclose(fid);
fprintf('WROTE %d rows, %d non-NaN\n', nc*nf, sum(~isnan(S.nuc_solidity(:))));
fprintf('DONE\n');

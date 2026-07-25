% Dump the MATLAB pipeline's mesh-derived metrics for a few live Jurkat cells, so
% the Python port can be compared against them. Read-only; writes nothing.

f = ['F:\FF\nucleus_live_cell\jurkat_nucleus_centrosome\GFP-Centrin_SiR-DNA\' ...
     'Control\cells\all_cells_together\prog_live_cells\' ...
     'Jurkats_live_Control_04142022_results.mat'];
S = load(f);
vars = fieldnames(S);
fprintf('VARS %s\n', strjoin(vars', ', '));

for k = 1:numel(vars)
    v = S.(vars{k});
    fprintf('VAR %s class=%s size=%s\n', vars{k}, class(v), mat2str(size(v)));
    if isstruct(v)
        fn = fieldnames(v);
        hits = fn(contains(fn, 'mesh') | contains(fn, 'nuc_height') | ...
                  contains(fn, 'sphericity') | contains(fn, 'SA'));
        fprintf('  FIELDS_OF_INTEREST %s\n', strjoin(hits', ', '));
    end
end
fprintf('DONE\n');

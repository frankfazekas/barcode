addpath(genpath('C:\Users\Upadhyaya_Lab\Code\TCell-3D-Morphodynamics\src'));

fprintf('MCPATH [%s]\n', mcpath('cgalsurf'));
fprintf('EXESUFF [%s]\n', getexeext);
fprintf('WHICH_V2S [%s]\n', which('v2s'));
fprintf('WHICH_MCPATH [%s]\n', which('mcpath'));
fprintf('WHICH_VOL2SURF [%s]\n', which('vol2surf'));
fprintf('WHICH_BINSURFACE [%s]\n', which('binsurface'));

R = 14; N = 40; c = (N-1)/2;
[i1, i2, i3] = ndgrid(0:N-1, 0:N-1, 0:N-1);
ball = ((i1-c).^2 + (i2-c).^2 + (i3-c).^2) <= R^2;
fprintf('CLASS %s NNZ %d\n', class(ball), nnz(ball));

newimg = zeros(size(ball)+2, class(ball));
newimg(2:end-1, 2:end-1, 2:end-1) = ball;
[lv, le] = binsurface(newimg >= 0.99);
pt = surfinterior(lv, le);
fprintf('CENT %.6f %.6f %.6f\n', pt(1), pt(2), pt(3));

% run cgalsurf by hand with fully explicit args
thres = 0.99 - 1e-4*0.99;
newdim = size(newimg);
brad = sum(newdim.*newdim)*2;
pre = [tempname '.inr']; post = [tempname '.off'];
saveinr(uint8(newimg), pre);
exe = [mcpath('cgalsurf') getexeext];
cmd = [' "' exe '" "' pre '" ' sprintf('%.16f %.16f %.16f %.16f %.16f %.16f %.16f %.16f %d ', ...
    thres, pt(1), pt(2), pt(3), brad, 30, 5, 5, 40000) ' "' post '" ' sprintf('%.0f %d', hex2dec('623F9A9E'), 50)];
fprintf('CMD %s\n', cmd);
[st, ~] = system(cmd);
fprintf('STATUS %d\n', st);
[n, e] = readoff(post);
fprintf('RAWOFF %d %d\n', size(n,1), size(e,1));
fprintf('DONE\n');

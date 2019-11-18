function out = fs_vel(params, precomp)
%% G = fundamental velocity solution in 2D (2x2 matrix)

etabar = params.eta + params.etaR;
r = precomp.r;
d = precomp.d;

rbar = precomp.rbar;
rbar2 = precomp.rbar2;

bk0 = precomp.bk(1);
bk1 = precomp.bk(2);
bk2 = precomp.bk(3);

out = (eye(2)*(-1+rbar*bk1 + rbar2*bk0) + (d*d')*(2-rbar2*bk2)/r^2)/...
    (2*pi*rbar2*etabar);

end

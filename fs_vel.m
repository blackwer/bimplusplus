function out = fs_vel(params, precomp)
%% G = fundamental velocity solution in 2D (2x2 matrix)

etabar = params.eta + params.etaR;
r = precomp.r;
d = precomp.d;

rbar = precomp.rbar;
rbar2 = precomp.rbar2;
delta = params.delta;

bk0 = precomp.bk0;
bk1 = precomp.bk1;
bk2 = precomp.bk2;

out = eye(2)*(-1+rbar*bk1 + rbar2*bk0)/...
    (2*pi*rbar2*etabar) + (d*d')*(2-rbar2*bk2)/...
    (2*pi*r^2*rbar2*etabar);

end

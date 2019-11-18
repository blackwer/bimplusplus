function out = fs_vel_p(params, t, d, r, rbar, rbar2, bk)
%% G' = derivative of G wrt s, depends on tangent at field, t

delta = params.delta;
eta = params.eta;
etaR = params.etaR;
lam = params.lam;

etabar = eta+etaR;

drds = dot(t,d/r);

bk0 = bk(1);
bk1 = bk(2);
bk2 = bk(3);

term1 = eye(2)*(2*r*(-1+rbar*bk1+rbar2*bk0)+r*rbar2*(bk0-rbar*bk1))*drds +...
    t*d'*(2-rbar2*bk2) + d*t'*(2-rbar2*bk2) + d*d'*rbar2*bk1*drds*lam;

term2 = 4*r*drds*(eye(2)*(-1+rbar*bk1 + rbar2*bk0) + (d*d')*(2-rbar2*bk2)/r^2);

out = (term1-term2)/(2*pi*etabar*r^4/delta^2);
end
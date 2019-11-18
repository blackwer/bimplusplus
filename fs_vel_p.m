function out = fs_vel_p(params, precomp, t)
%% G' = derivative of G wrt s, depends on tangent at field, t

delta = params.delta;
eta = params.eta;
etaR = params.etaR;

etabar = eta+etaR;

r = precomp.r;
d = precomp.d;
lam = precomp.lam;
rbar = precomp.rbar;
rbar2 = precomp.rbar2;
drds = dot(t,d/r);

bk0 = precomp.bk(1);
bk1 = precomp.bk(2);
bk2 = precomp.bk(3);

%delta_ij = mod(i + j+1,2)
term1= zeros(2,2);
for i = 1:2
    for k = 1:2
        term1(i,k) = mod(i+k+1,2)*(2*r*(-1+rbar*bk1+rbar^2*bk0)+r*rbar^2*(bk0-rbar*bk1))*drds...
            + t(i)*d(k)*(2-rbar^2*bk2) + d(i)*t(k)*(2-rbar^2*bk2) + d(i)*d(k)*rbar^2*bk1*drds*lam;
    end
end
term2 = 8*pi*etabar*r*rbar2*drds*fs_vel(params, precomp);

out = (term1-term2)/(2*pi*etabar*r^4/delta^2);
end
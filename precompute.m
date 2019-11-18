function precomp = precompute(params, source, field)
%PRECOMPUTE compute re-used quantities for bim_test
precomp.d = field-source;
precomp.r = norm(precomp.d);
precomp.lam = 1/params.delta;
precomp.rbar = precomp.r*precomp.lam;
precomp.rbar2 = precomp.rbar*precomp.rbar;
precomp.bk0 = besselk(0, precomp.rbar);
precomp.bk1 = besselk(1, precomp.rbar);
precomp.bk2 = besselk(2, precomp.rbar);
precomp.bk3 = besselk(3, precomp.rbar);
end
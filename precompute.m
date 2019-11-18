function precomp = precompute(params, source, field)
%PRECOMPUTE compute re-used quantities for bim_test
precomp.d = field-source;
precomp.r = norm(precomp.d);
precomp.lam = 1/params.delta;
precomp.rbar = precomp.r*precomp.lam;
precomp.rbar2 = precomp.rbar*precomp.rbar;
precomp.bk = besselk([0, 1, 2, 3], precomp.rbar);
end
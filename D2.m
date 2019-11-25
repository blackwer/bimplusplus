%% spectral second derivative on periodic interval d(num)/d(den)
% assumes "denom" is uniformly spaced
function dfdx = D2(numer,denom)

N = length(denom);
k = 2*pi./(N*(denom(2)-denom(1)))*[0 1:(N-1)/2 -(N-1)/2:-1]; %weird order
F = fft(numer);%.*exp(-10*(abs(k)/N).^25);
%F(abs(F) < 1e-13) = 0; %Krasny filter
dfdx = ifft(-k.*k.*F);

end
function T = fs_trac(params, precomp)
%% FS_TRAC fundamental traction solution in 2D assuming normal = radial vector (2x2 matrix)
%% only pressure + shear stress components
r = precomp.r;
d = precomp.d;
lam = precomp.lam;
rbar = precomp.rbar;

%delta_ij = mod(i + j+1,2)

bk1 = precomp.bk1;
bk2 = precomp.bk2;
bk3 = precomp.bk3;

T = zeros(2,2,2); %T_ijk
for i = 1:2
    for j = 1:2
        for k = 1:2
            T(i,j,k) = mod(i+j+1,2)*d(k)*(4-rbar^2-2*rbar^2*bk2)/(2*pi*lam^2*r^4)...
                + (mod(i+k+1,2)*d(j)+mod(j+k+1,2)*d(i))*(4-2*rbar^2*bk2-rbar^3*bk1)/(2*pi*lam^2*r^4)...
                + d(i)*d(j)*d(k)*(-8+rbar^3*bk3)/(pi*lam^2*r^6);
        end
    end
end

end
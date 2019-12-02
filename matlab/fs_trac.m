function T = fs_trac(params, d, r, rbar, bk)
%% FS_TRAC fundamental traction solution in 2D assuming normal = radial vector (2x2 matrix)
%% only pressure + shear stress components
lam = params.lam;

%delta_ij = mod(i + j+1,2)

bk1 = bk(2);
bk2 = bk(3);
bk3 = bk(4);

coeff1 = (4-rbar^2-2*rbar^2*bk2)/(2*pi*lam^2*r^4);
coeff2 = (4-2*rbar^2*bk2-rbar^3*bk1)/(2*pi*lam^2*r^4);
coeff3 = (-8+rbar^3*bk3)/(pi*lam^2*r^6);

T = zeros(2,2,2); %T_ijk
for i = 1:2
    for j = 1:2
        for k = 1:2
            T(i,j,k) = mod(i+j+1,2)*d(k)*coeff1...
                + (mod(i+k+1,2)*d(j)+mod(j+k+1,2)*d(i))*coeff2...
                + d(i)*d(j)*d(k)*coeff3;
        end
    end
end

end
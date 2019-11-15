%% Brinkman flow in perturbed unit circle
%% arclength = alpha parameterization for unit circle
%% nonaxisymmetric case, AB2 time stepping
%% WIP

function [] = bim_test4()

%% -------- startup --------
clc; close all; 
tic;

%% --------- physical parameters (let Omega = 1) --------
global etaR; etaR = 1; %rotational viscosity
global eta; eta = 1; %shear viscosity
global etaO; etaO = 1; %odd viscosity
global G; G = 10; %substrate drag (big Gamma)
global delta; delta = sqrt((eta+etaR)/G); %BL length scale
global gam; gam = 0.01; %line tension (little gamma)

%% -------- numerical parameters --------
global N; N = 2^7-1; %number of points on curve
dt = 0.001;
t = 0; %time
t_max = 0.1;
filename = '2Dflow_v8';
window_x = 1.2; %window sizes
window_y = 1.2;
soltol = 1e-12;

%% -------- initialize boundary (periodic BCs) ---------
alpha = linspace(0,2*pi,N+1); alpha = alpha(1:end-1); %fixed dimensionless parameterization
eps = 0.1; %perturbation amplitude
mp = 4; %perturbation mode
x = cos(alpha) + eps*sin(mp*alpha).*cos(alpha);
y = sin(alpha) + eps*sin(mp*alpha).*sin(alpha);

%% -------- geometric quantities --------
dxda = D(x,alpha); %dx/d(alpha)
dyda = D(y,alpha); %dy/d(alpha)
L_n = trapzp(sqrt(dxda.^2+dyda.^2));
s_i = linspace(0,L_n,N+1); s_i = s_i(1:end-1); %arclength coordinate (uniform spacing)
a_i = zeros(size(s_i));
for i = 1:length(a_i)
    a_i(i) = fzero(@(a) s_i(i) - integral(@(ap) sqrt(mp^2*eps^2*cos(ap*mp).^2+(1+eps*sin(ap*mp)).^2),0,a),0.1);
end
x_i = cos(a_i) + eps*sin(mp*a_i).*cos(a_i);
y_i = sin(a_i) + eps*sin(mp*a_i).*sin(a_i);
positions_n = [x_i; y_i]; %2xN matrix of coordinates of equally spaced (in s) positions
%dsda = sqrt(D(x,alpha).^2+D(y,alpha).^2);  %ds/d(alpha)

%% -------- (x,y) -> (theta,L) --------
x_ip = D(x_i,alpha); x_ipp = D2(x_i,alpha);
y_ip = D(y_i,alpha); y_ipp = D2(y_i,alpha);
tangents_n = [(x_ip*2*pi/L_n); (y_ip*2*pi/L_n)];
normals_n = [-(y_ip*2*pi/L_n); (x_ip*2*pi/L_n)];
kappa_n = (x_ip.*y_ipp - y_ip.*x_ipp)./(sqrt(x_ip.^2+y_ip.^2)).^3;
theta_n = atan2(y_ip(1),x_ip(1)) + L_n/(2*pi)*cumtrapz(alpha,kappa_n) ...
          + (2*pi - L_n/(2*pi)*trapzp(kappa_n));
dthda_n = L_n/(2*pi)*kappa_n;

area_n = trapzp(x_i.^2+y_i.^2)/2; % integral of r dr dtheta
% cm_n = [2*pi/3/area_n/L_n*trapzp(x_i.*(x_i.^2+y_i.^2)) ; 
%         2*pi/3/area_n/L_n*trapzp(y_i.*(x_i.^2+y_i.^2))];
% ind = trapzp(kappa_n)*L_n/(2*pi); %equals 2*pi

%% -------- given curve, solve linear system for flow --------
uv_np1 = inteqnsolve(positions_n,tangents_n,normals_n,L_n,soltol,zeros(2*N,1)); %2Nx1 matrix

%% -------- plots --------
%plot(theta_n); hold on; grid on;
% plot(alpha,(u'.*cos(theta_n)+v'.*sin(theta_n))); hold on; grid on; 
% plot(alpha,(u'.*sin(theta_n)-v'.*cos(theta_n)));
% vtrue = delta*etaR*besseli(1,1/delta)/(eta*besseli(2,1/delta)+etaR*besseli(0,1/delta))
% plot(alpha,vtrue*cos(alpha),'o-');
fig = figure(); set(gca,'FontSize',18); set(gcf,'color','w');
plot([x_i x_i(1)],[y_i y_i(1)],'bo-','LineWidth',2); axis equal; grid on; hold on; box on;
axis([-window_x window_x -window_y window_y]); axis equal;
plot(sqrt(area_n/pi)*[ cos(alpha) cos(alpha(1))],sqrt(area_n/pi)*[sin(alpha) sin(alpha(1))],'r-');
title('t = 0 , A/A_0 = 1');

%% -------- save plot to movie --------
VW = VideoWriter([filename '.avi']);
VW.FrameRate = 30;
open(VW);
frame = getframe(fig);
writeVideo(VW,frame);

for j = 1
%% -------- first time step --------
%% compute U
U_n = sum(normals_n.*[uv_np1(1:2:end-1) uv_np1(2:2:end)]');
q2 = quiver(positions_n(1,:),positions_n(2,:),U_n.*normals_n(1,:),U_n.*normals_n(2,:)); q2.AutoScale = 'off';
T_n = cumtrapz(alpha,dthda_n.*U_n) - alpha/(2*pi)*trapzp(dthda_n.*U_n);

%% update theta and L (Euler forward for 1 step)
L_np1 = L_n - dt*trapzp(dthda_n.*U_n);
theta_np1 = theta_n + dt*(2*pi/L_n)*(D(U_n,alpha) + dthda_n.*T_n);
tangents_np1 = [cos(theta_np1); sin(theta_np1)];
normals_np1 = [-sin(theta_np1); cos(theta_np1)];

%% update 1 point, then use (x,y) = integral of tangent
X_np1 = positions_n(:,1) + dt*(U_n(1)*normals_n(:,1) + T_n(1)*tangents_n(:,1));
x_np1 = X_np1(1) + L_np1/(2*pi)*cumtrapz(alpha,cos(theta_np1)) - L_np1/(2*pi)*trapzp(cos(theta_np1));
y_np1 = X_np1(2) + L_np1/(2*pi)*cumtrapz(alpha,sin(theta_np1)) - L_np1/(2*pi)*trapzp(sin(theta_np1));
positions_np1 = [x_np1; y_np1];

%% conserved quantities
area_np1 = trapzp(x_np1.^2+y_np1.^2)/2; % integral of r dr dtheta
%cm_np1 = [2*pi/3/area_np1/L_np1*trapzp(x_np1.*(x_np1.^2+y_np1.^2)) ;
%          2*pi/3/area_np1/L_np1*trapzp(y_np1.*(x_np1.^2+y_np1.^2))];

%% using new positions, compute new curvature and therefore d(theta)/da
x_ip = D(x_i,alpha); x_ipp = D2(x_i,alpha);
y_ip = D(y_i,alpha); y_ipp = D2(y_i,alpha);
dthda_np1 = L_np1/(2*pi)*(x_ip.*y_ipp - y_ip.*x_ipp)./(sqrt(x_ip.^2+y_ip.^2)).^3;

%clf; 
t = t + dt;
plot(x_np1,y_np1,'ro-','LineWidth',2); hold on; grid on;
axis([-window_x window_x -window_y window_y]); axis equal;
plot(sqrt(area_n/pi)*[cos(alpha) cos(alpha(1))],sqrt(area_n/pi)*[sin(alpha) sin(alpha(1))],'r-'); %circle
title([' t = ' num2str(t) ' , ' 'A/A_0 = ' num2str(area_np1/area_n)]);

end

%% -------- time stepping -------
uv_n = uv_np1;
while t < t_max
    
    %% update positions (simple AB2 scheme, do we need higher order?)
    % dx/dt = u(x,y,t), dy/dt = v(x,y,t), yn+1 = yn + dt/2*(3*f(tn,yn)-f(tn-1,yn-1))
    
    %% compute U and T
    uv_np2 = inteqnsolve(positions_np1,tangents_np1,normals_np1,L_np1,soltol,2*uv_np1-uv_n); %2Nx1 matrix
    U_np1 = sum(normals_np1.*[uv_np2(1:2:end-1) uv_np2(2:2:end)]'); 
    T_np1 = cumtrapz(alpha,dthda_np1.*U_np1) - alpha/(2*pi)*trapzp(dthda_np1.*U_np1);    
    
    %% update theta and L
    L_np2 = L_np1 - dt/2*(3*trapzp(dthda_np1.*U_np1) - trapzp(dthda_n.*U_n)); %AB2
    theta_np2 = theta_np1 + dt/2*(3*(2*pi/L_np2)*(D(U_np1,alpha) + dthda_np1.*T_np1) - (2*pi/L_np1)*(D(U_n,alpha) + dthda_n.*T_n));
    tangents_np2 = [cos(theta_np2); sin(theta_np2)];
    normals_np2 = [-sin(theta_np2); cos(theta_np2)];
    
    %% integrate tangent to get X(alpha)
    X_np2 = positions_np1(:,1) + dt/2*(3*U_np1(1)*normals_np2(:,1) - U_n(1)*normals_np1(:,1)); %AB2
    x_np2 = X_np2(1) + L_np2/(2*pi)*cumtrapz(alpha,cos(theta_np2)) - L_np2/(2*pi)*trapzp(cos(theta_np2));
    y_np2 = X_np2(2) + L_np2/(2*pi)*cumtrapz(alpha,sin(theta_np2)) - L_np2/(2*pi)*trapzp(sin(theta_np2));
    positions_np2 = [x_np2; y_np2];
    
    %% conserved quantities
    area_np2 = trapzp(x_np2.^2+y_np2.^2)/2; % integral of r dr dtheta
%     cm_np2 = [2*pi/3/area_np2/L_np1*trapzp(x_np2.*(x_np2.^2+y_np2.^2)) ; 
%               2*pi/3/area_np2/L_np1*trapzp(y_np2.*(x_np2.^2+y_np2.^2))];
    
    %% calculate new curvature
    x_ip = D(x_np2,alpha); x_ipp = D2(x_np2,alpha);
    y_ip = D(y_np2,alpha); y_ipp = D2(y_np2,alpha);
    dthda_np2 = L_np2/(2*pi)*(x_ip.*y_ipp - y_ip.*x_ipp)./(sqrt(x_ip.^2+y_ip.^2)).^3;
    
    %% plot & save image
    clf;
    plot([x_np2 x_np2(1)],[y_np2 y_np2(1)],'bo-','LineWidth',2); grid on; 
    axis([-window_x window_x -window_y window_y]); axis equal; hold on;
    plot(sqrt(area_n/pi)*[cos(alpha) cos(alpha(1))],sqrt(area_n/pi)*[sin(alpha) sin(alpha(1))],'r-');
    %plot(cm_np2(1),cm_np2(2),'r.','MarkerSize',10);
    title([' t = ' num2str(t) ' , ' 'A/A_0 = ' num2str(area_np2/area_n)]);
    q2 = quiver(positions_np2(1,:),positions_np2(2,:),U_np1.*normals_np2(1,:),U_np1.*normals_np2(2,:)); q2.AutoScale = 'off';
    frame = getframe(fig);
    writeVideo(VW,frame);

    %% change n and n-1 timestep info
    dthda_n = dthda_np1; U_n = U_np1; T_n = T_np1; uv_n = uv_np2;
    positions_np1 = positions_np2; normals_np1 = normals_np2; tangents_np1 = tangents_np2;
    L_np1 = L_np2; dthda_np1 = dthda_np2; theta_np1 = theta_np2; uv_np1 = uv_np2;
    t = t + dt;
    
    
end


%% -------- save last known positions to txt file --------
fileID = fopen([filename '.txt'],'w');
fprintf(fileID,'%12.8f\n',L_np1);
fprintf(fileID,'%12.8f\n',theta_np1');
fprintf(fileID,'%12.8f\n',dthda_np1');

%% -------- cleanup --------
close(VW);
fclose(fileID);
toc;

end

%% G = fundamental velocity solution in 2D (2x2 matrix)
function out = fs_vel(source,field)
global delta;
global eta;
global etaR;

etabar = eta+etaR;
r = norm(field-source);
d = field-source;

out = eye(2)*(-1+1/delta*r*besselk(1,1/delta*r) + 1/delta^2*r^2*besselk(0,1/delta*r))/...
    (2*pi*1/delta^2*etabar*r^2) + (d*d')*(2-1/delta^2*r^2*besselk(2,1/delta*r))/...
    (2*pi*1/delta^2*etabar*r^4);

end

%% G' = derivative of G wrt s, depends on tangent at field, t
function out = fs_vel_p(source,field,t)

global delta;
global eta;
global etaR;

etabar = eta+etaR;

r = norm(field-source);
d = field-source;
lam = 1/delta;
rbar = r*lam;
drds = dot(t,d/r);

%delta_ij = mod(i + j+1,2)
term1= zeros(2,2);
for i = 1:2
    for k = 1:2
        term1(i,k) = mod(i+k+1,2)*(2*r*(-1+rbar*besselk(1,rbar)+rbar^2*besselk(0,rbar))+r*rbar^2*(besselk(0,rbar)-rbar*besselk(1,rbar)))*drds...
            + t(i)*d(k)*(2-rbar^2*besselk(2,rbar)) + d(i)*t(k)*(2-rbar^2*besselk(2,rbar)) + d(i)*d(k)*rbar^2*besselk(1,rbar)*drds*lam;
    end
end
term2 = 8*pi*etabar/delta^2*r^3*drds*fs_vel(source,field);

out = (term1-term2)/(2*pi*etabar*r^4/delta^2); 


end

%% T = fundamental traction solution in 2D assuming normal = radial vector (2x2 matrix)
%% only pressure + shear stress components
function out = fs_trac(source,field)

global delta;

r = norm(field-source);
d = field-source;
lam = 1/delta;
rbar = r*lam;

%delta_ij = mod(i + j+1,2)

T = zeros(2,2,2); %T_ijk
for i = 1:2
    for j = 1:2
        for k = 1:2
            T(i,j,k) = mod(i+j+1,2)*d(k)*(4-rbar^2-2*rbar^2*besselk(2,rbar))/(2*pi*lam^2*r^4)...
                + (mod(i+k+1,2)*d(j)+mod(j+k+1,2)*d(i))*(4-2*rbar^2*besselk(2,rbar)-rbar^3*besselk(1,rbar))/(2*pi*lam^2*r^4)...
                + d(i)*d(j)*d(k)*(-8+rbar^3*besselk(3,rbar))/(pi*lam^2*r^6);
        end
    end
end

out = T;


end

%% notes about FFT in MATLAB
%1. output is returned out of natural order, use fftshift
%2. periodic interval, start and end points should not same in general
%3. use N odd for simplicity?
%4. FFT defined without constants, IFFT defined with factor of 1/n

%% spectral derivative on periodic interval d(num)/d(den)
% assumes "denom" is uniformly spaced
function dfdx = D(numer,denom)

N = length(denom);
k = 2*pi./(N*(denom(2)-denom(1)))*[0 1:(N-1)/2 -(N-1)/2:-1]; %weird order
F = fft(numer);%.*exp(-10*(abs(k)*4/N).^25);
%F(abs(F) < 1e-13) = 0; %Krasny filter
dfdx = ifft((1i*k).*F);

end

%% spectral second derivative on periodic interval d(num)/d(den)
% assumes "denom" is uniformly spaced
function dfdx = D2(numer,denom)

N = length(denom);
k = 2*pi./(N*(denom(2)-denom(1)))*[0 1:(N-1)/2 -(N-1)/2:-1]; %weird order
F = fft(numer);%.*exp(-10*(abs(k)/N).^25);
%F(abs(F) < 1e-13) = 0; %Krasny filter
dfdx = ifft(-k.*k.*F);

end

%% solve the boundary integral equation given a closed curve
function vel = inteqnsolve(positions,tangents,normals,L,toler,vel_prev)

global N;
global etaO;
global etaR;
global gam;

sys = zeros(2*N,2*N);
RHS = zeros(2*N,1);
for m = 1:N
    temp = zeros(2,1);
    for n = 1:N
        if m ~= n  %off diagonal terms are pv integrals (skip over singularity)
            T = fs_trac(positions(:,m),positions(:,n));
            M1 = [normals(1,n)*T(1,1,1) normals(1,n)*T(2,1,1); %already transposed
                  normals(1,n)*T(1,1,2) normals(1,n)*T(2,1,2)];
            M2 = [normals(2,n)*T(1,2,1) normals(2,n)*T(2,2,1); %already transposed
                  normals(2,n)*T(1,2,2) normals(2,n)*T(2,2,2)];
            sys(2*m-1:2*m,2*n-1:2*n) =  -(L/N)*(M1+M2) - (L/N)*...
                fs_vel_p(positions(:,m),positions(:,n),tangents(:,n))*...
                2*(-1*etaO*eye(2) + etaR*[0, 1;-1, 0]);
            temp = temp + fs_vel_p(positions(:,m),positions(:,n),tangents(:,n))...
                *(-1*etaR*positions(:,n)-gam*tangents(:,n)); %% outward vs inward n confusion
        else       %diagonal terms
            sys(2*m-1:2*m,2*n-1:2*n) = (1/2)*eye(2);
        end
    RHS(2*m-1:2*m) = (L/N)*temp;
    end
end

[vel,~] = gmres(sys,RHS,[],toler,N,[],[],vel_prev); %two outputs suppresses printing

end

%% periodic trapezoid rule over full period w.r.t. alpha = s*2pi/L
function out = trapzp(vec)
global N;

out = 2*pi/N*sum(vec);

end

















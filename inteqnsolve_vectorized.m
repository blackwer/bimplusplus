function vel = inteqnsolve_vectorized(params,positions,tangents,normals,L,toler,vel_prev)
%% INTEQNSOLVE solve the boundary integral equation given a closed curve

N = params.N;
eta0 = params.eta0;
etaR = params.etaR;
gam = params.gam;

sys = zeros(2*N,2*N);
sys_low = zeros(N, 2*N);
sys_high = zeros(N, 2*N);
RHS = zeros(N, 2);
%% can now use parfor on this loop, but it's not really much faster...
%% 2 threads gives me about a 20% speedup. 12 threads ~2x.
%% numthreads = 4;
%% parfor (m = 1:N, numthreads)
for m = 1:N
    temp = zeros(2,1);
    sys_row = zeros(2, 2*N);
    
    source = positions(:,m);
    d = positions-source;
    r = sqrt(sum(d.^2,1));
    rbar = r*params.lam;
    rbar2 = rbar.*rbar;
    bk0 = besselk(0, rbar);
    bk1 = besselk(1, rbar);
    bk2 = besselk(2, rbar);
    bk3 = besselk(3, rbar);
 
    for n = 1:N
        if m ~= n  %off diagonal terms are pv integrals (skip over singularity)
            precomp.d = d(:,n);
            precomp.r = r(n);
            precomp.rbar = rbar(n);
            precomp.rbar2 = rbar2(n);
            precomp.bk = [bk0(n), bk1(n), bk2(n), bk3(n)];
            
            T = fs_trac(params, precomp);
            M1 = [normals(1,n)*T(1,1,1) normals(1,n)*T(2,1,1); %already transposed
                  normals(1,n)*T(1,1,2) normals(1,n)*T(2,1,2)];
            M2 = [normals(2,n)*T(1,2,1) normals(2,n)*T(2,2,1); %already transposed
                  normals(2,n)*T(1,2,2) normals(2,n)*T(2,2,2)];

            G_prime = fs_vel_p(params, precomp, tangents(:,n));

            sys_row(:,2*n-1:2*n) =  -(L/N)*(M1+M2) - (L/N)*...
                G_prime*2*(-1*eta0*eye(2) + etaR*[0, 1;-1, 0]);
            temp = temp + ...
                G_prime*(-1*etaR*positions(:,n)-gam*tangents(:,n)); %% outward vs inward n confusion
        else       %diagonal terms
            sys_row(:,2*n-1:2*n) = (1/2)*eye(2);
        end
    end
    RHS(m,:) = (L/N)*temp;
    sys_low(m,:) = sys_row(1,:);
    sys_high(m,:) = sys_row(2,:);
end
RHS = reshape(RHS',[2*N,1]);

for m = 1:N
    sys(2*m-1,:) = sys_low(m,:);
    sys(2*m,:) = sys_high(m,:);
end

[vel,~] = gmres(sys,RHS,[],toler,N,[],[],vel_prev); %two outputs suppresses printing

end

function [ ] = plotresultsvec( u_knots, v_knots,P0,P1,P2,P3,X,z, Err)

[np k] = size(X);

u_n_basf = length(u_knots)-3;
v_n_basf = length(v_knots)-3;


nu = 60;
nv = 60;

Zsurf = zeros(nu,nv);
Xsurf = zeros(nu,nv);
Ysurf = zeros(nu,nv);

% Compute X & Y control points


[ X_coor,Y_coor] = compute_control_points( u_knots, v_knots, P0,P1,P2,P3);

x = X_coor(:);
y = Y_coor(:);


% Compute fine grid in X & Y & Z for draw

uf = spalloc(u_n_basf,nu,3*nu); 
vf = spalloc(v_n_basf,nv,3*nv); 

for j =1:nu
    u = (j-1)/(nu-1);
    uf(:,j) = splinebasevec(u_knots,u,0);
end

for k =1:nv
    v = (k-1)/(nv-1);
    vf(:,k) = splinebasevec(v_knots,v,0);
end

for k =1:nv
        Zsurf(:,k) = kron(vf',uf(:,k)') * z; 
        Xsurf(:,k) = kron(vf',uf(:,k)') * x;
        Ysurf(:,k) = kron(vf',uf(:,k)') * y;
end



surf(Xsurf,Ysurf,Zsurf);


%%plot original points

% hold on
% for k=1:np
%     if X(k,4) ~=0
%         plot3(X(k,1),X(k,2),X(k,3),'.k','MarkerSize',25);
%     end
% end

%%

% hold off
% 
%  figure
%  
%  Xsurferr = kron(xv(2:u_n_basf-1)',ones(v_n_basf-2,1));
%  Ysurferr = kron(yv(2:v_n_basf-1),ones(u_n_basf-2,1)');
% 
%  
% surf(Xsurferr,Ysurferr,Err);

end


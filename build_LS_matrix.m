function [ B,Interv ] = build_LS_matrix( u_knots,v_knots, Xp )
u_n_basf = length(u_knots)-3;
v_n_basf = length(v_knots)-3;
[np k] = size(Xp); % k unused
B =spalloc(np,u_n_basf*v_n_basf,9*np);
Interv = zeros(np,2);
for j = 1:np
    [uf, k] = splinebasevec(u_knots,Xp(j,1),0);
    [vf, i] = splinebasevec(v_knots,Xp(j,2),0);
    Interv(j,:) = [k,i];
    B(j,:) = kron(vf',uf');
end
end


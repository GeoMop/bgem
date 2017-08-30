function [ X_coor,Y_coor ] = compute_patch_edges( u_knots, v_knots, P0,P1,P2,P3)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
u_n_bound = length(u_knots)-4;
v_n_bound = length(v_knots)-4;


X_coor = zeros(u_n_bound,v_n_bound);
Y_coor = zeros(u_n_bound,v_n_bound);

for i =1:u_n_bound
    u = u_knots(i+2);
    for j =1:v_n_bound
        v =  v_knots(j+2);
        P = u * (v * P2 + (1-v) * P1) + (1-u) * ( v * P3 + (1-v) * P0) ; 
        X_coor(i,j) = P(1);
        Y_coor(i,j) = P(2);   
    end
end


end


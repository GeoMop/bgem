function [ A ] = build_reg_matrix( u_knots,v_knots, P0,P1,P2,P3,nnzA )

u_n_basf = length(u_knots)-3;
v_n_basf = length(v_knots)-3;
u_n_inter = length(u_knots) - 5
v_n_inter = length(v_knots) - 5

    a = P3 - P2;
    b = P0 - P1;
    c = P1 - P2;
    d = P0 - P3;
    
A =spalloc(u_n_basf*v_n_basf,u_n_basf*v_n_basf,nnzA);

% Qpoints = [0 (1/2 - 1/sqrt(20)) (1/2 + 1/sqrt(20)) 1];
% weights = [1/6 5/6 5/6 1/6];

Qpoints = [-0.90618 -0.538469 0  0.538469  0.90618] /2 + 0.5;
weights = [0.236927 0.478629 0.568889 0.478629 0.236927];



n_points = length(Qpoints);

u_point_val = spalloc(u_n_basf,u_n_inter*n_points,u_n_inter*n_points*3);
ud_point_val = spalloc(u_n_basf,u_n_inter*n_points,u_n_inter*n_points*3);
q_u_point = zeros(u_n_inter*n_points,1);

n = 0;
for i = 1:u_n_inter
    us = u_knots(i+2);
    uil = u_knots(i+3)- u_knots(i+2); 
    for k = 1:n_points
        up = us + uil*Qpoints(k);
        n = n+1;
        q_u_point(n) = up;
        u_point_val(:,n) = splinebasevec(u_knots,up,0,i);
        ud_point_val(:,n) = splinebasevec(u_knots,up,1,i);  
    end
end

%%%

v_point_val =  spalloc(v_n_basf,v_n_inter*n_points,v_n_inter*n_points*3);
vd_point_val = spalloc(v_n_basf,v_n_inter*n_points,v_n_inter*n_points*3);
q_v_point = zeros(v_n_inter*n_points,1);

n = 0;
for i = 1:v_n_inter
    vs = v_knots(i+2);
    vil = v_knots(i+3)- v_knots(i+2);
    for k = 1:n_points
        vp = vs + vil*Qpoints(k);
        n = n+1;
        q_v_point(n) = vp;
        v_point_val(:,n) = splinebasevec(v_knots,vp,0,i);
        vd_point_val(:,n) = splinebasevec(v_knots,vp,1,i);
    end
end

%%%

for i= 1:v_n_inter
    for k = 1:n_points
        v_point = v_point_val(:,(i-1)*n_points+k);
        vd_point =vd_point_val(:,(i-1)*n_points+k);
        for l =1:u_n_inter
            for m = 1:n_points
                u_point = u_point_val(:,(l-1)*n_points+m);
                ud_point = ud_point_val(:,(l-1)*n_points+m);
                vd = kron(vd_point,u_point);
                ud = kron(v_point,ud_point);
                v = q_v_point((i-1)*n_points +k);
                u = q_u_point((l-1)*n_points +m);
                J = det([v * a + (1 - v) * b, u * c + (1 - u) * d]);
                A = A + J * weights(m)*weights(k)*(kron(ud,ud') +kron(vd,vd'));
            end
        end
    end
end

end

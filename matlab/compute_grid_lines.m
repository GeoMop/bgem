function [ GX,GY ] = compute_grid_lines(u_knots,v_knots, P0,P1,P2,P3 )

u_n_grid = length(u_knots)-2;
v_n_grid = length(v_knots)-2;

GX = zeros(u_n_grid,v_n_grid);
GY = zeros(u_n_grid,v_n_grid);


for i =1:u_n_grid
    u = u_knots(i+2);
    for j =1:v_n_grid
        v = v_knots(j+2);
        P = u * (v * P2 + (1-v) * P1) + (1-u) * ( v * P3 + (1-v) * P0) ; 
        GX(i,j) = P(1);
        GY(i,j) = P(2);   
    end
end

end


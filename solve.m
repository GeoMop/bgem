function [ Xp ] = solve( Xp,P0,P1,P2,P3 )

[np k] = size(Xp);
Pi = [P0 P1 P2 P3];

%%% Compute local coordinates and drop all

A = P3 - P2;
B = P0 - P1;
C = P1 - P2;
D = P0 - P3;

for j=1:np
    
    P = [Xp(j,5); Xp(j,6)];
    
    u =  Xp(j,1);
    v =  Xp(j,2);
    
    uv = [u;v];
    
    iters = 10;
    
    for i = 1:iters
        u  = uv(1);
        v  = uv(2);
        
        %         J = [ v * A(1) + (1-v) * B(1), u * C(1) + (1 - u) * D(1);
        %             v * A(2) + (1-v) * B(2), u * C(2) + (1 - u) * D(2)];
        J = [ v * A + (1-v) * B, u * C + (1 - u) * D];
        Fxy = P - u * (v * P2 + (1-v) * P1) - (1-u) * ( v * P3 + (1-v) * P0);
        uv = [u;v] - inv(J) * Fxy;
    end
    Xp(j,1) = uv(1);
    Xp(j,2) = uv(2);
end
end


function [ Xp X] = transform( X,P0,P1,P2,P3 )

[np k] = size(X);

Pi = [P0 P1 P2 P3];

%%% Compute normalized normals

Ni = zeros(2,4);


Ni(:,1) = Pi(:,1) - Pi(:,4);
nt =  Ni(1,1);
Ni(1,1) = -Ni(2,1);
Ni(2,1) = nt;
Ni(:,1) = Ni(:,1)/norm(Ni(:,1));



for i =2:4
    Ni(:,i) = Pi(:,i) - Pi(:,i-1);
    nt =  Ni(1,i);
    Ni(1,i) = -Ni(2,i);
    Ni(2,i) = nt;
    Ni(:,i) = Ni(:,i)/norm(Ni(:,i));
end

%%% Compute local coordinates and drop all 

h = 0;
for j=1:np
    
    P = [X(j,1); X(j,2)];
    
    u =  (P - Pi(:,1))' * Ni(:,1) / ( (P - Pi(:,1))' * Ni(:,1) + (P - Pi(:,3))' * Ni(:,3)    ) ;
    v =  (P - Pi(:,1))' * Ni(:,2) / ( (P - Pi(:,1))' * Ni(:,2) + (P - Pi(:,4))' * Ni(:,4)    ) ;
%     P - Pi(:,1)
%     P - Pi(:,4)
%     Ni(:,2)
%     Ni(:,4)
%     (P - Pi(:,1))' * Ni(:,2)
%     (P - Pi(:,4))' * Ni(:,4) 
%     pause
    
    if (u >= 0) && (u <= 1) && (v >= 0) && (v <= 1)
        h = h+1;
        Xp(h,1) = u;
        Xp(h,2) = v;
        Xp(h,3:4) = X(j,3:4);
        Xp(h,5:6) = X(j,1:2);
    end
end

end


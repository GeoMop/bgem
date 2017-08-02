function [ bool ] = boundary_point(uv,iujv,u_knots,v_knots)
bool = [-1,-1];

uint = [u_knots(iujv(1)+2), u_knots(iujv(1)+3)];
vint = [v_knots(iujv(2)+2), v_knots(iujv(2)+3)];
uv;


if uint(1) == 0
    if abs(uv(1)-0)<eps
        bool(1) = 0;
    end
else
    if abs(uv(1)-uint(1))<eps
        bool(1) = 1;
    end 
end

if uint(2) == 1
    if abs(uv(1)-1)<eps
        bool(1) = 0;
    end
else
    if abs(uv(1)-uint(2))<eps
        bool(1) = 1;
    end
end

%%%%

if vint(1) == 0
    if abs(uv(2)-0)<eps
        bool(2) = 0;
    end
else
    if abs(uv(2)-vint(1))<eps
        bool(2) = 1;
    end 
end

if vint(2) == 1
    if abs(uv(2)-1)<eps
        bool(2) = 0;
    end
else
    if abs(uv(2)-vint(2))<eps
        bool(1) = 1;
    end
end

   
   %pause
    
end


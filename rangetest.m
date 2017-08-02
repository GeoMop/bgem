function [ test, uvt ] = rangetest( uvt,ui,vi,u2i,v2i,tol )


test = 0;

du = [uvt(1)-ui(1), ui(2)- uvt(1)];
dv = [uvt(2)-vi(1), vi(2)- uvt(2)];

if length(v2i) == 1
d2p = [uvt(3)-u2i(1), u2i(2)- uvt(3)];
pi = u2i;
end

if length(u2i) == 1
d2p = [uvt(3)-v2i(1), v2i(2)- uvt(3)];
pi = v2i;
end

%dv = [uvt(2)-vi(1), vi(2)- uvt(2)];


for i =1:2
    if (du(i) < -tol) 
        uvt(1) = ui(i);
    end
end

for i =1:2
    if (dv(i) < -tol)
        uvt(2) = vi(i);
    end
end

for i =1:2
    if (d2p(i) < -tol) 
        uvt(3) = pi(i);
    end
end


if (uvt(1)>=ui(1)) && (uvt(1)<=ui(2))
    if (uvt(2)>=vi(1)) && (uvt(2)<=vi(2))
        if ((uvt(3)>=pi(1)) && (uvt(3)<=pi(2)))
            test = 1;
        end
    end
end


end


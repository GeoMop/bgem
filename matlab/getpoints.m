function [ X ] = getpoints()

nu = 36;
nv = 26;
h = 0;
X = zeros(nu*nv,4);

for i=1:nu
    for j = 1:nv
        h = h+1;
        x = 2*(i-1)/(nu-1);
        y = 2*(j-1)/(nv-1);
        X(h,1) = x;
        X(h,2) = y;
        %X(h,3) = 0.3*cos(3*pi*exp(-x))*cos(3*pi*y^1.2)*cos(3*pi*sqrt(y^2+x^2)); 
        %X(h,3) = x^2+y^2 + 1; 
        X(h,3) = (2-x^2)+(2-y^2) + 1; 
        X(h,4) = 1;
    end
end

end


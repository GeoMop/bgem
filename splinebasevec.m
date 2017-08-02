function [ f, k ] = splinebasevec( T,t,order,varargin)

%  T - knot vector
%  k - index of basis function, k = 1,...,length(T)-3
%  t - parameter t \in [0,1]
%  f - vector of basis functions values
%  k - kth interval

n = length(T);

f = spalloc(n-3,1,3);

optargin = length(varargin);

if optargin == 0
    k = find_int( T,t );
elseif optargin == 1
    k = varargin{1};
%     if (T(k) <= t+2)  && (T(k+3) >= t)
%         disp('OK')
%         [T(k+2), t,T(k+3)]
%      else
%          disp('PROBLEM')
%          return
%      end
if (T(k) <= t+2)  && (T(k+3) >= t)
    %         disp('OK');
    [T(k+2), t,T(k+3)];
else
    disp('PROBLEM')
    pause
    return
end
end


% for k = 1:n-5
%     if(t>=T(k+2) && t<=T(k+3))
%        % k_int = k;
%         break;
%     end
% end
%
% if k ~=k2-2
%  [k k2-2]
%  pause
% end

tk1 = T(k+1);
tk2 = T(k+2);
tk3 = T(k+3);
tk4 = T(k+4);

d31 = tk3-tk1;
d32 = tk3-tk2;
d42 = tk4-tk2;

d3t = tk3-t;
dt1 = t-tk1;
dt2 = t-tk2;
d4t = tk4-t;

d31d32 = d31*d32;
d42d32 = d42*d32;

% f(k) = (tk3-t)^2/((tk3-tk1)*(tk3-tk2));
% f(k+1)= (t-tk1)*(tk3 -t)/((tk3-tk1)*(tk3-tk2)) + (t-tk2)*(tk4 -t)/((tk4-tk2)*(tk3-tk2));
% f(k+2) = (t-tk2)^2/((tk4 - tk2)*(tk3-tk2));

if order == 0
    f(k) = d3t^2/d31d32;
    f(k+1)= dt1*d3t/d31d32 + dt2*d4t/d42d32;
    f(k+2) = dt2^2/d42d32;
elseif order ==1
    f(k) = -2*d3t/d31d32;
    f(k+1)= (d3t - dt1)/d31d32 + (d4t - dt2)/d42d32;
    f(k+2) = 2*dt2/d42d32;
end

end
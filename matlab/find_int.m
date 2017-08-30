function [ k ] = find_int( T,t )

n = length(T);
mn = 3;
mx = n-2;
est = 3+floor(t*(n -5));

if t == T(mn)
    k = mn-2;
elseif t == T(mx)
    k = mx-3;
end

if t>= T(est)
    mn =est;
elseif t < T(est) % =
    mx = est;
end 

s = max(ceil(log2(mx-mn)),1);

for p=1:s+1
    if t < T(mn+1) % =
        k = mn-2;
        break
        
    elseif t>T(mx-1) % =
        k = mx-3;
        break
    end   
    mid = mn + floor((mx-mn)/2);
    %mn
    %mx
    if mid ~= mn
        if t< T(mid)
            mx = mid;
        elseif t>= T(mid)
            mn = mid;
        end  
    else
        k = mn-2;
        break
    end
end

% T
% [T(mn) T(mx)]
% if (t~=0) && (t~=1)
%     if (t>=T(k) && t<=T(k+1) )
%         [T T >t]
%         t
%         k
%         disp('problem')
%         pause
%     end
% end

end


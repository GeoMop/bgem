function [ Err ] = get_errors(err,Interv,u_n,v_n )

n = length(err);
Err = zeros(v_n-2,u_n-2);
for j =1:n
    Err(Interv(j,2),Interv(j,1)) = Err(Interv(j,2),Interv(j,1)) + err(j);
end

%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here


end


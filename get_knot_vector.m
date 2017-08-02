function [ knots ] = get_knot_vector( n_basf )


knots = zeros(n_basf+3,1);
knots(3:n_basf+1) = linspace(0,1,n_basf-1);
knots(n_basf+2:n_basf+3) = 1;


end


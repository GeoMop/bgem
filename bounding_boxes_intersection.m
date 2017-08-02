function [ its, n_its] = bounding_boxes_intersection( BB_X,BB_Y,Z_coor, BB_Xs,BB_Ys,Zs_coor)


[u_n_intervs, v_n_intervs,~] = size(BB_X);
[us_n_intervs, vs_n_intervs,~] = size(BB_Xs);


its = zeros(u_n_intervs-1, v_n_intervs-1);
n_its = zeros(u_n_intervs-1, v_n_intervs-1);

for k=1:u_n_intervs-1
    for l=1:v_n_intervs-1
        for m=1:us_n_intervs-1
            for o=1:vs_n_intervs-1
                sp_i = (m-1)*(us_n_intervs-1) + o;
                
                mnX = min(min(BB_X(k:k+1,l:l+1)));
                mxX = max(max(BB_X(k:k+1,l:l+1)));
                mnXs = min(min(BB_Xs(m:m+1,o:o+1)));
                mxXs = max(max(BB_Xs(m:m+1,o:o+1)));
                
                mnY = min(min(BB_Y(k:k+1,l:l+1)));
                mxY = max(max(BB_Y(k:k+1,l:l+1)));
                mnYs = min(min(BB_Ys(m:m+1,o:o+1)));
                mxYs = max(max(BB_Ys(m:m+1,o:o+1)));
                
                mnZ = min(min(Z_coor(k:k+2,l:l+2)));
                mxZ = max(max(Z_coor(k:k+2,l:l+2)));
                mnZs = min(min(Zs_coor(m:m+2,o:o+2)));
                mxZs = max(max(Zs_coor(m:m+2,o:o+2)));
                
                
                if (( (mnX >=  mnXs) && (mnX <=  mxXs)) || ...
                        ((mxX >=  mnXs) && (mxX <=  mxXs)) ) == 1
                    
                    if (( (mnY >=  mnYs) && (mnY <=  mxYs)) || ...
                            ((mxY >=  mnYs) && (mxY <=  mxYs)) ) == 1
                        if (( (mnZ >=  mnZs) && (mnZ <=  mxZs)) || ...
                                ((mxZ >=  mnZs) && (mxZ <=  mxZs)) ) == 1
                            
                            
                            n_its(k,l) = n_its(k,l) + 1;
                            its(k,l,n_its(k,l)) = sp_i;
                            
                        end
                    end
                end
            end
        end
    end
end


end


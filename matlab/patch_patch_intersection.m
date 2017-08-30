function [ uvt,pt,conv ] = patch_patch_intersection( uvt, ui,vi,u2i,v2i,u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l)

pt =zeros(3,1);
conv =0;
tol = 1e-4;
tol2 = 1e-6;

if length(u2i) == 1
    [u2f, ~] = splinebasevec(us_knots,u2i,0,m);
end
if length(v2i) == 1
    [v2f, ~] = splinebasevec(vs_knots,v2i,0,o);
end


for i=1:nit
    [uf, ~] = splinebasevec(u_knots,uvt(1),0,k);
    [vf, ~] = splinebasevec(v_knots,uvt(2),0,l);
    [ufd, ~] = splinebasevec(u_knots,uvt(1),1,k);
    [vfd, ~] = splinebasevec(v_knots,uvt(2),1,l);
      
    if length(u2i) == 1
        [v2f, ~] = splinebasevec(vs_knots,uvt(3),0,o);
        [v2fd, ~] = splinebasevec(vs_knots,uvt(3),1,o);
        dXYZp2 = (kron(v2fd',u2f')*Xs)';
    end
    if length(v2i) == 1
        [u2f, ~] = splinebasevec(us_knots,uvt(3),0,m);
        [u2fd, ~] = splinebasevec(us_knots,uvt(3),1,m);
        dXYZp2 = (kron(v2f',u2fd')*Xs)';
    end

    dXYZu1 = (kron(vf',ufd')*X)'; %
    dXYZv1 = (kron(vfd',uf')*X)'; %
    
    J = [dXYZu1 dXYZv1 -dXYZp2];
    
    deltaXYZ = (kron(vf',uf') * X)' - (kron(v2f',u2f') * Xs)';
    uvt = uvt- J\deltaXYZ;
    %if i~=nit
        [test,uvt] = rangetest(uvt,ui,vi,u2i,v2i,0.0);
    %end
end

[test,uvt] = rangetest(uvt,ui,vi,u2i,v2i,tol2);
if test == 1
    [uf, ~] = splinebasevec(u_knots,uvt(1),0,k);
    [vf, ~] = splinebasevec(v_knots,uvt(2),0,l);
   
    if length(u2i) == 1
        [v2f, ~] = splinebasevec(vs_knots,uvt(3),0,o);
    end
    if length(v2i) == 1
        [u2f, ~] = splinebasevec(us_knots,uvt(3),0,m);
    end  
    
    deltaXYZ = (kron(vf',uf') * X)' - (kron(v2f',u2f') * Xs)';
    dist = norm(deltaXYZ);
end

if test == 1
    if dist <= tol
        pt =kron(vf',uf')*X;
        conv =1;
    end
else
    uvt = zeros(3,1);
end

end


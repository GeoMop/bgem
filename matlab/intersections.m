
% spline_surf_vec
 %close all
% % %
% plotresultsvec(u_knots, v_knots,P0,P1,P2,P3,X,z,Err)
% hold on
% plotresultsvec(us_knots, vs_knots,P0s,P1s,P2s,P3s,Xs,zs,Errs)
hold on


%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Compute intersections
%%%%%%%%%%%%%%%%%%%%%%%%%

% u_n_basf = length(u_knots)-3;
% v_n_basf = length(v_knots)-3;
% us_n_basf = length(us_knots)-3;
% vs_n_basf = length(vs_knots)-3;

u_n_intervs = u_n_basf - 2;
v_n_intervs = v_n_basf - 2;
us_n_intervs = us_n_basf - 2;
vs_n_intervs = vs_n_basf - 2;

u_n_grid  = u_n_basf+1;
v_n_grid  = v_n_basf+1;
us_n_grid  = us_n_basf+1;
vs_n_grid  = vs_n_basf+1;

% u_n_intervs = u_n_basf - 2;
% v_n_intervs = v_n_basf - 2;
% us_n_intervs = us_n_basf - 2;
% vs_n_intervs = vs_n_basf - 2;

% Compute X & Y grid intersections

%%% Grid Boundary
[ GX,GY ] = compute_grid_lines(u_knots,v_knots, P0,P1,P2,P3 );
[ GXs,GYs ] = compute_grid_lines(us_knots,vs_knots, P0s,P1s,P2s,P3s );

%%% Grid centers
[ X_coor,Y_coor] = compute_control_points( u_knots, v_knots, P0,P1,P2,P3);
[ Xs_coor,Ys_coor] = compute_control_points( us_knots, vs_knots, P0s,P1s,P2s,P3s);

%%% Compute Bounding Boxes
Z_coor  = vec2mat( z,u_n_basf,v_n_basf );
Zs_coor  = vec2mat( zs,us_n_basf,vs_n_basf );

[patch_bound_X,patch_bound_Y] = compute_patch_edges( u_knots, v_knots, P0,P1,P2,P3);
[patch_bound_Xs,patch_bound_Ys] = compute_patch_edges( us_knots, vs_knots, P0s,P1s,P2s,P3s);

% [ BB_X,BB_Y,BB_Z ] = compute_bounding_box( X_coor,Y_coor,Z_coor, u_n_intervs,v_n_intervs);
% [ BB_Xs,BB_Ys,BB_Zs ] = compute_bounding_box( Xs_coor,Ys_coor,Zs_coor, us_n_intervs,vs_n_intervs);

%%% Bonding boxes intersections
[isec,n_isec] = bounding_boxes_intersection( patch_bound_X,patch_bound_Y,Z_coor,patch_bound_Xs,patch_bound_Ys,Zs_coor);


x = X_coor(:);
y = Y_coor(:);
xs = Xs_coor(:);
ys = Ys_coor(:);

Xs = [xs ys zs];

X = [x y z];
nt = 6; % number of main lines
n_points = 0;
ninter =  zeros(u_n_intervs,v_n_intervs);
for k=1:u_n_intervs
    us1 = u_knots(k+2);
    ue1 = u_knots(k+3);
    u1_c =(us1 + ue1)/2;
    ui = [us1 ue1] ;
    for l=1:v_n_intervs
        vs1 = v_knots(l+2);
        ve1 = v_knots(l+3);
        v1_c = (vs1  + ve1)/2;
        vi = [ vs1 ve1];        
         s=0;
        if n_isec(k,l) ~= 0
            for p =1: n_isec(k,l)
                
                m = ceil(isec(k,l,p) / us_n_intervs);
                o = isec(k,l,p) - (m-1)*us_n_intervs;
                sp_i = (m-1)*(us_n_intervs) + o;
                % v2 fixed
                u2i = [us_knots(m+2) us_knots(m+3)];
                v_knot = linspace(vs_knots(o+2),vs_knots(o+3),nt);            
                for h =1:length(v_knot)
                    u2_c = (us_knots(m+2) + us_knots(m+3))/2;
                    v2i = v_knot(h);
                    nit = 10;                    
                    uvu2v2 = [u1_c;v1_c;u2_c];
                    [ uvu2v2,ptl,conv ] = patch_patch_intersection( uvu2v2, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l);
                    if conv ~= 0
                        s = s+1;
                        plot3(ptl(1),ptl(2),ptl(3),'.','MarkerSize',50)
                        n_points = n_points +1;
                        point(n_points,:) = [uvu2v2',v_knot(h),ptl(1),ptl(2),ptl(3),k,l,m,o];
                    end 
                end

                % u2 fixed
                v2i = [vs_knots(o+2) vs_knots(o+3)];
                u_knot = linspace(us_knots(m+2),us_knots(m+3),nt);
                for h =1:length(u_knot)
                    v2_c = (vs_knots(o+2) + vs_knots(o+3))/2;
                    u2i = u_knot(h);
                    nit = 10;
                    uvu2v2 = [u1_c;v1_c;v2_c];
                    [ uvu2v2,ptl,conv ] = patch_patch_intersection( uvu2v2, ui,vi,u2i,v2i, u_knots, v_knots, X,us_knots, vs_knots, Xs, nit ,m,o,k,l);
                    if conv ~= 0
                        s = s+1;
                        plot3(ptl(1),ptl(2),ptl(3),'.','MarkerSize',50)
                        n_points = n_points +1;
                        point(n_points,:) = [uvu2v2(1:2)',u_knot(h),uvu2v2(3),ptl(1),ptl(2),ptl(3),k,l,m,o];
                    end
                end
                ninter(k,l) = s;
            end
        end
    end
end

 ninter
%[isec,n_isec] = bounding_boxes_intersection2( patch_bound_X,patch_bound_Y,Z_coor,patch_bound_Xs,patch_bound_Ys,Zs_coor);
%[isec2,n_isec2] = bounding_boxes_intersection2( patch_bound_Xs,patch_bound_Ys,Zs_coor,patch_bound_X,patch_bound_Y,Z_coor);

% for m=1:us_n_intervs
%     us2 = us_knots(m+2);
%     ue2 = us_knots(m+3);
%     u2_c = (us2+ue2)/2;
%     ui2 = [us2 ue2] ;
%     for o=1:vs_n_intervs
%         vs2 = vs_knots(o+2);
%         ve2 = vs_knots(o+3);
%         v2_c = (vs2+ve2)/2;
%         vi2 = [ vs2 ve2];
%         s = 0;
%         if n_isec2(m,o) ~= 0
%             for p =1: n_isec2(m,o)
%                 k = ceil(isec2(m,o,p) / u_n_intervs);
%                 l = isec2(m,o,p) - (k-1)*u_n_intervs;
%                 sp_i = (k-1)*(u_n_intervs) + l;
% %                 [sp_i isec2(m,o,p)]
% %                 pause
%                 t0 = 0.5;
%                 
%                 % v fixed
%                 v_knot = linspace(v_knots(l+2),v_knots(l+3),nt);
%                 for h =1:length(v_knot)
%                     u_val = linspace(u_knots(k+2),u_knots(k+3),3);
%                     v_val = v_knot(h);
%                     XYZ = get_span(u_knots,v_knots,X,u_val,v_val);
%                     nit = 6;
%                     uvt0 = [u2_c;v2_c;t0];
%                     [ uvtl, ptl ,conv] = compute_initial_guess( uvt0, ui2,vi2,us_knots, vs_knots, Xs, XYZ, nit,m,o);
%                     if conv ~= 0
%                         s = s+1;
%                         plot3(ptl(1),ptl(2),ptl(3),'.','MarkerSize',50)
%                     end
%                 end
%                 
%                 % u fixed
%                 u_knot = linspace(u_knots(k+2),u_knots(k+3),nt);
%                 for h =1:length(u_knot)
%                     u_val = u_knot(h);
%                     v_val = linspace(v_knots(l+2),v_knots(l+3),3);
%                     XYZ = get_span(u_knots,v_knots,X,u_val,v_val);
%                     nit = 6;
%                     uvt0 = [u2_c;v2_c;t0];
%                     [ uvtl, ptl ,conv] = compute_initial_guess( uvt0, ui2,vi2,us_knots, vs_knots, Xs, XYZ, nit,m,o);
%                     if conv ~= 0
%                         s = s+1;
%                         plot3(ptl(1),ptl(2),ptl(3),'.','MarkerSize',50)
%                     end
%                 end
%                 
%                 ninter(m,o) = ninter(m,o) + s;
%             end
%         end
%     end
% end

%t = toc

%cas = sum(sum(ninter))/t

ninter
 plot3(GXs,GYs,5*ones(us_n_basf+1,vs_n_basf+1))
 plot3(GYs,GXs,5*ones(us_n_basf+1,vs_n_basf+1))

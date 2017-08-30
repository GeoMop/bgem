%close all

spline_surf_vec
intersections

 plot3(GXs,GYs,5*ones(us_n_basf+1,vs_n_basf+1));
 hold on
 plot3(GYs,GXs,5*ones(us_n_basf+1,vs_n_basf+1));

 for i=1:m
      plot3(point(i,5),point(i,6),point(i,7),'.','MarkerSize',50);
 end
 
%pointx = point;
pointb = point;


%%% sort intersections by patches (1st patch)

 sp_i = (pointb(:,10)-1)*(us_n_intervs-1) + pointb(:,11);

[ssp_i idx] = sort(sp_i);

m = length(sp_i); %m-out



pointb = pointb(idx,:);

%%% Compute intersectioned patches
patch_int(1,2) = 1;
patch_int(1,1) = ssp_i(1);
%a = sp_i(1);
different = 1;
for i =2:m
    if ssp_i(i) == patch_int(different,1)%a
        patch_int(different,2) = patch_int(different,2)  +1;
        continue
    else
        %a = sp_i(i);
        different = different +1;
        patch_int(different,1) = ssp_i(i);
        patch_int(different,2) = 1;
    end
end
 

different;



%return



coinf = zeros(m,2);


a = 1; %??

%%% detect intersection point types 
% -1 - interion
%  0 - global boundary
%  1 - patch boundary

for j=1:m
        coi = boundary_point(pointb(j,3:4),pointb(j,10:11),us_knots,vs_knots);
        coinf(j,:) = coi; % type of interrsection (u,v) % 1D
end

out = zeros(different,1);
point_type = zeros(m,1); % 2D

offset = 0;
for j=1:different
    for i = 1:patch_int(j,2)
        
        if coinf(i+offset,1) > -1 || coinf(i+offset,2) > -1  % number of boundary points for patch
            out(j) = out(j) +1;
        end 
        
        % define intersection type
        if coinf(i+offset,1) == -1 && coinf(i+offset,2) == -1  % internal 
            point_type(i+offset) = -1;
        elseif   coinf(i+offset,1) == 0 || coinf(i+offset,2) == 0 % global boundary
            point_type(i+offset) = 0;
        elseif   coinf(i+offset,1) == 1 || coinf(i+offset,2) == 1 % patch internal boundary
            point_type(i+offset) = 1;
        end
        
        
        
    end
    offset = offset + patch_int(j,2);
end
 

% number of outputs



% % Sort points in patch
%   offset = 0;  
%   for j=1:different
%       
%       
%       patch_points= offset:offset+patch_int(j,2);
%       
%       boundg = find(point_type(patch_points) == 0)
%       boundi = find(point_type(patch_points) == 1)
%       bound = [boundg , boundi];
%       
%       l = lenght(bound)
%       if l >2
%           disp('more then 2 boundary points')
%       end
%       
%       dist = zeros(patch_int(j,2));
%       
%       for k = 1: offset+patch_int(j,2)
%           for i = 1: offset+patch_int(j,2)
%               
%               dist(i) = norm(pointsb(patch_points(k),1:2)- pointsb(patch_points(i),1:2))
%               i+offset
%               
%           end
%           
%       end
%       offset = offset + patch_int(j,2);
%   end
%

% Sort points in surface (create components)

  offset = 0;  
  bound = find(point_type == 0);
  pointb = [pointb coinf point_type];
  [a,b] = size(pointb);
  splinepoint = pointb;
  splinepoint = swaplines(splinepoint,1,bound(1));
 
  for j=1:m-1
      dist = 1/eps * ones(m,1);
      for k=j+1:m
      dist(k) = norm(splinepoint(j,1:2) - splinepoint(k,1:2));  
      end
      [a b] = min(dist);
      splinepoint = swaplines(splinepoint,b,j+1);
  end
  
  % Sort intersection on patches
  boundg = find(point_type == 0);
  boundl = find(point_type == 1);
  bound = [boundg;boundl]
  
%   offset =0;
%   for k=1:different
%       boundg = find(point_type(1+offset:patch_int(different,2)+offset)== 0);
%       boundl = find(point_type(1+offset:patch_int(different,2)+offset)== 1);
%       bound = [boundg;boundl]
%       %pause
%       pointb = swaplines(pointb,bound(1)+offset,1+offset)
%       for l=1:patch_int(k,2)-1
%           dist = 1/eps * ones(patch_int(k,2)-l,1);
%           for j=1:patch_int(k,2)-l
%               dist(j) = norm(splinepoint(l,1:2) - splinepoint(l+j,1:2));
%           end
%           [a b] = min(dist);
%           splinepoint = swaplines(splinepoint,b+offset,l+1+offset);
%           
%       end
%   offset = offset + patch_int(k,2)
%   end
  
  
 
  
  %%%% Remove duplicite points ()
  
  matrixpoints = ones(m,1); 
  a = 0;
  for i=1:m
%       if splinepoint(i,14) == 1
%           a = 1;
%       end
      if splinepoint(i,14)*a == 1
          matrixpoints(i) = 0;
          a = 0;
      elseif splinepoint(i,14) == 1
          a =1;
      else
          a = 0;
      end
  end
  
  
% Interpolate points
%  
% 
% pointb = pointb(1:m-out,:)
% 
% mm = m - out;
% 
% %return
% 
mr = sum(matrixpoints);
XYZ = zeros(mr,3);
deltas = zeros(mr,1);
j = 0;

for i=1:m
    if matrixpoints(i)==1
        j = j+1;
        XYZ(j,:) =splinepoint(i,5:7);
        if j~=1
            deltas(j) = norm(XYZ(j,:)-XYZ(j-1,:));
        end
    end
end

Deltas = zeros(mr,1);

for i=2:mr
    Deltas(i)=sum(deltas(1:i));
end

pointspersegment=4;
nbasf = ceil(mr/pointspersegment);
t_knots = get_knot_vector(nbasf+2);
t = Deltas/Deltas(mr);
X = zeros(mr,nbasf+2);
for i=1:mr
    
    [tf, ~] = splinebasevec(t_knots,t(i),0);
    X(i,:) = tf;
end

sol = X\XYZ;
X*sol - XYZ;

%X'*X\X'*XYZ

ns = 100;
td = linspace(0,1,ns);
for i=1:ns
    PT(i,:)=splinebasevec(t_knots,td(i),0)'*sol;
    
end
plot3(PT(:,1),PT(:,2),PT(:,3),'r','LineWidth',3);
%
%
%
%
%
%
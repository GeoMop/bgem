
clc
clear all
close all


u_n_basf = 15;
v_n_basf = 15;
u_knots = get_knot_vector(u_n_basf);
v_knots = get_knot_vector(v_n_basf);

%%% base test

% basetestvec(v_knots);
% return
%%% patch boundary

% P0 = [0.2; 0.3];
% P1 = [1.2; 0.5];
% P2 = [1.5; 2];
% P3 = [0.9; 1.7];

P0 = [0.0; 0.0];
P1 = [2.0; 0.0];
P2 = [2.0; 2.0];
P3 = [0.0; 2.0];

%%% Reduce points to be approximatex (i.e., points which are in convex hull of the points P0,P1,P2,P3)

X = getpoints();
[Xp X] = transform( X,P0,P1,P2,P3 );
[np k] = size(Xp);
[ Xp ] = solve( Xp,P0,P1,P2,P3 );

%%% Construction of the matrix


%interv ??
[B, Interv ] = build_LS_matrix( u_knots,v_knots, Xp);

W = sparse(diag(Xp(:,4)));

g = Xp(:,3);
b = B'*W*g;

C = B'*W*B;
nnzC = nnz(C);


A = build_reg_matrix( u_knots,v_knots, P0,P1,P2,P3,nnzC);

nC = norm(full(C));
nA = norm(full(A));
r = norm(full(C))/  norm(full(A));
%r = 0.0;

% [q r] = qr(B);
% 
% 
% z = r\(q'*g)

S = C+0.01*r*A;

z = pcg(S,b,1e-12,500);

%%% Solution

% Direct Solver
% C = B'*W*B;%+A;
% b = B'*W*g;
% z = C\b;


% Errors
Err = get_errors(abs(W*B*z-g),Interv,u_n_basf,v_n_basf);


%%% PLOT results

plotresultsvec(u_knots, v_knots,P0,P1,P2,P3,X,z,Err)

%return

%%%%%%%%%%%%%%%%%%
%%% Second Surface
%%%%%%%%%%%%%%%%%%


us_n_basf = 10;
vs_n_basf = 10;
us_knots = get_knot_vector(us_n_basf);
vs_knots = get_knot_vector(vs_n_basf);

%%% patch boundary

% P0s = [0.4; 0.2];
% P1s = [1.5; 0.1];
% P2s = [1.1; 1.8];
% P3s = [0.6; 1.7];

P0s = [0.0; 0.0];
P1s = [2.0; 0.0];
P2s = [2.0; 2.0];
P3s = [0.0; 2.0];


%%% Reduce points to be approximatex (i.e., points which are in convex hull of the points P0,P1,P2,P3)

Xs = getpoints2();
[Xps Xs] = transform( Xs,P0s,P1s,P2s,P3s );
[nps ks] = size(Xps);
[ Xps ] = solve( Xps,P0s,P1s,P2s,P3s );

%%% Construction of the matrix

%interv ??
[Bs, Intervs ] = build_LS_matrix( us_knots,vs_knots, Xps);

Ws = sparse(diag(Xps(:,4)));

gs = Xps(:,3);
bs = Bs'*Ws*gs;

Cs = Bs'*Ws*Bs;
nnzCs = nnz(Cs);


As = build_reg_matrix( us_knots,vs_knots, P0s,P1s,P2s,P3s,nnzCs);

nCs= norm(full(Cs));
nAs = norm(full(As));
rs = norm(full(Cs))/  norm(full(As));
%rs = 0.0;

Ss = Cs+0.0010*rs*As;

zs = pcg(Ss,bs,1e-14,500);
%zs = Ss\bs;

%%% Solution

% Direct Solver
% C = B'*W*B;%+A;
% b = B'*W*g;
% z = C\b;


% Errors
Errs = get_errors(abs(Ws*Bs*zs-gs),Intervs,us_n_basf,vs_n_basf);


%%% PLOT results
hold on
plotresultsvec(us_knots, vs_knots,P0s,P1s,P2s,P3s,Xs,zs,Errs)
hold off


%intersections

%
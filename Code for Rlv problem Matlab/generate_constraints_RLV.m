function dy=generate_constraints_RLV(time_breaks,point_each,R,S,V,Gamma,M,Theta,Omega,T,Beta)
% Here time breaks is an array of points [t0,t1,...,tn,tau_f]
% points each is the number of points in each segment.
% Total number of collocation points become sum(point_each)-(number of time points -2)
% therefore size of H will be (no. of collocation points)*1

%g0,R0,m0,Cd,S_ref,Isp,l_com,I_z
global R0 g0 m0 s0 v0 h0 theta0 gamma0 omega0

Cd=0.5;

S_ref=10.75;
I_z=3346393;
l_com=9.778; % centre of mass distance
Isp=300;

t_ref=32;
coll_points=sum(point_each)-(length(time_breaks)-2);
if(length(R)~=coll_points)
    disp('Error in number of collocation points. Check!');
end

k=1;

for l=1:length(time_breaks)-1
    r=R(k:point_each(l)+k-1);
    s=S(k:point_each(l)+k-1);
    v=V(k:point_each(l)+k-1);
    gamma=Gamma(k:point_each(l)+k-1);
    m=M(k:point_each(l)+k-1);
    theta=Theta(k:point_each(l)+k-1);
    omega=Omega(k:point_each(l)+k-1);
    t=T(k:point_each(l)+k-1);
    beta=Beta(k:point_each(l)+k-1);
    
    
    Dr=0.5*1.225*((v*v0).^2).*Cd*S_ref/(m0*g0);
    
    
    X=[r,s,v,gamma,m,theta,omega];
    F=[v.*sin(gamma)*(v0*t_ref/R0),v.*cos(gamma)*(v0*t_ref/s0),(g0*t_ref/v0)*((-t.*cos(beta-gamma+theta)-Dr)./m-sin(gamma)./(r.^2)),(g0*t_ref/v0)*((-t.*sin(beta-gamma+theta))./(m.*v)-cos(gamma)./((r.^2).*v)),-t/Isp*t_ref,omega*t_ref,-t.*sin(beta)*l_com/I_z*t_ref*m0*g0 ];

    [D,~]=cheb(-1,1,point_each(l)-1);
    if l==1
        %disp(size(D*X));
        %disp(size(0.5*(time_breaks(l+1)-time_breaks(l))*F));
        %L(1:cum_points(l),2)=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;
        L=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;
    else
        %L(cum_points(l-1)+1:cum_points(l),2)=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;
        L=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;
    end
    k=k+point_each(l)-1;
    if l==1
        dy=L;
        
    else
        dy=vertcat(L,dy);
        
    end
end

end
function dy=generate_constraints_beta_der(time_breaks,point_each,Beta,T)
% Here time breaks is an array of points [t0,t1,...,tn,tau_f]
% points each is the number of points in each segment. 
% Total number of collocation points become sum(point_each)-(number of time points -2)
% therefore size of H will be (no. of collocation points)*1
   
%g0,R0,m0,Cd,S_ref,Isp,l_com,I_z 

t_ref=32;
coll_points=sum(point_each)-(length(time_breaks)-2);
if(length(Beta)~=coll_points)
    disp('Error in number of collocation points. Check!');
end

k=1;

for l=1:length(time_breaks)-1
   beta=Beta(k:point_each(l)+k-1);
   TT=T(k:point_each(l)+k-1);
    [D,~]=cheb(-1,1,point_each(l)-1);
    
    L=D*beta<=0.5*(time_breaks(l+1)-time_breaks(l))*10*pi/180*t_ref;
    dl=-D*beta<=0.5*(time_breaks(l+1)-time_breaks(l))*10*pi/180*t_ref;
    L=vertcat(L,dl);
    dm=D*TT<=0.5*(time_breaks(l+1)-time_breaks(l))*25;
    dn=-D*TT<=0.5*(time_breaks(l+1)-time_breaks(l))*25;
    L=vertcat(L,dm);
    L=vertcat(L,dn);
    %L(cum_points(l-1)+1:cum_points(l),2)=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;    
    %L=D*X==0.5*(time_breaks(l+1)-time_breaks(l))*F;
       k=k+point_each(l)-1;
    if l==1
        dy=L;
        
    else
        dy=vertcat(L,dy);
        
    end
end

end
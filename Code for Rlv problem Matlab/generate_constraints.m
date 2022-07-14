function dy=generate_constraints_RLV(time_breaks,point_each,H,V,U)
% Here time breaks is an array of points [t0,t1,...,tn,tau_f]
% points each is the number of points in each segment. 
% Total number of collocation points become sum(point_each)-(number of time points -2)
% therefore size of H will be (no. of collocation points)*1
g=1.5;
coll_points=sum(point_each)-(length(time_breaks)-2);
if(length(H)~=coll_points)
    disp('Error in number of collocation points. Check!');
end

k=1;

for l=1:length(time_breaks)-1
   
    X=[H(k:point_each(l)+k-1),V(k:point_each(l)+k-1)];
    F=[V(k:point_each(l)+k-1),-g+U(k:point_each(l)+k-1)];
    
    disp(size(X));
    disp(size(F));
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
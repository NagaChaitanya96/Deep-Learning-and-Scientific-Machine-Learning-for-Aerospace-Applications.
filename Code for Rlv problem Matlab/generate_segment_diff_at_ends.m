function dy=generate_segment_diff_at_ends(time_breaks,point_each,R,S,V,Gamma,M,Theta,Omega,T,Beta)

k=1;

for l=1:length(time_breaks)-1
   %r=R(k:point_each(l)+k-1);
   %s=S(k:point_each(l)+k-1);
   %v=V(k:point_each(l)+k-1);
   %gamma=Gamma(k:point_each(l)+k-1);
   %m=M(k:point_each(l)+k-1);
   %theta=Theta(k:point_each(l)+k-1);
   %omega=Omega(k:point_each(l)+k-1);
   t=T(k:point_each(l)+k-1);
   beta=Beta(k:point_each(l)+k-1);
   
   [D,~]=cheb(time_breaks(l),time_breaks(l+1),point_each(l)-1);
   %der=D*[r s v gamma m theta omega t beta];
   der=D*[t beta];
   der_first=der(1,:);
   if l==2
      dy=der_first-der_end==0;
   end
   if l>2
       ll=der_first-der_end==0;
       dy=[dy;
           ll];
   end
   
   der_end=der(end,:);
   k=k+point_each(l)-1;
end

end
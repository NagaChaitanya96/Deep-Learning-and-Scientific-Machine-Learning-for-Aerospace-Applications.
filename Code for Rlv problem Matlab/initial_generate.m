function [h_init,v_init]=initial_generate(t_points,t_end,point_each,h_0,v_0)
Time=[t_points,t_end];
time_ref=[];
for i=1:length(point_each)
    [~,ref]=cheb(-1,1,point_each(i)-1);
    temp=0.5*(Time(i+1)-Time(i))*ref+0.5*(Time(i+1)+Time(i));
   if i==1
    time_ref=temp;
   else
    time_ref=[time_ref temp(2:end)];
   end
end
disp(size(time_ref));
tt=linspace(0,t_end,100);
hh=linspace(0,h_0,100);
vv=linspace(0,v_0,100);
h_init=interp1(tt,hh,time_ref);
v_init=interp1(tt,vv,time_ref);
end
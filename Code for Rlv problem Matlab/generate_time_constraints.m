function dy=generate_time_constraints(T_breaks,tau_f,point_each)
if length(point_each)<2
    disp('Error cannot assign constraints');
end
dy=-T_breaks(1)<=0;
for i=1:length(point_each)-2
        dy=vertcat(dy,T_breaks(i)<=T_breaks(i+1));
end
dy=vertcat(dy,T_breaks(end)<=tau_f);

end
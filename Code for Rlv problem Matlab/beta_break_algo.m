function [T_knots,N_knots]=beta_break_algo(beta,time,rho,N_min,N_max)

ind=find(beta(2:end-1)>rho);
T_knots=[];
ind_array=[];
if isempty(ind)
    [~,indd]=max(beta(2:end-1));
    indd=indd+1;
    T_knots=[T_knots time(indd)];
    %disp(indd);
    ind_array=[ind_array indd];
else
    ind=ind+1; % to offset the error caused by 2:end-1, essentially indd=indd+1
    for j=1:length(ind)
        if ind(j)~=1 && ind(j)~=length(time)
            if beta(ind(j))>beta(ind(j)-1) && beta(ind(j))>beta(ind(j)+1)
                T_knots=[T_knots time(ind(j))];
                ind_array=[ind_array ind(j)];
            end
        end
    end
end
T_knots=sort(T_knots);
%N_knots=N_min*ones(1,length(T_knots)+1);
ind_array=sort(ind_array);
ind_array=[1 ind_array length(beta)];
N_knots=diff(ind_array)+1;
for jj=1:length(N_knots)
   if N_knots(jj)>N_max
       N_knots(jj)=N_max;
   end
   if N_knots(jj)<N_min
       N_knots(jj)=N_min;
   end
    
end

% if max(beta)==beta(end)
%     T_knots=time(floor(length(beta)*0.75));
%     N_knots=[N_max,N_min];
% end

end


function [df,time]=cheb(a,b,N)

for i=0:N
    for j=0:N
       if i==0 && j==0
          df(i+1,j+1)=2/(b-a)*(2*(N)^2 + 1)/6; 
       end
        
       if i==N && j==N
          df(i+1,j+1)=-2/(b-a)*(2*(N)^2 + 1)/6;  
       end
       
       if i==j && i~=0 && i~=N
          t_q = cos(pi*(i)/N);
          df(i+1,j+1)= -1/(b-a)*t_q/(1-t_q^2);   
       end
       
       if i~=j  
         
           if i>=1 && i<=N-1
              c_p=1; 
           end
           if j>=1 && j<=N-1
              c_q=1; 
           end
           if i==0 || i==N
              c_p=2; 
           end
    
           if j==0 || j==N
              c_q=2; 
           end
           t_p=cos(pi*i/N);
           t_q=cos(pi*j/N);
           df(i+1,j+1)=2/(b-a)*c_p/c_q*((-1)^(i+j))/(t_p-t_q);
       end
       
       
    end
    
end

df=-df;
time=0.5*(b-a)*cos(pi*(0:N)/N)+(a+b)/2;
time=flip(time);
end
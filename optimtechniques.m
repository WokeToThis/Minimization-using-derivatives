syms x y gk
f =  x^5 * exp(-x^2-y^2)

 fsurf(f,[-3 3])
 fcontour(f,[-3 3])


%%
% 
xk=1
yk =-1
% %subs(f,[x y],[xk yk])
% gradx = subs(dfx,[x y],[xk yk])
% grady = subs(dfy,[x y],[xk yk])
% 
% g =@(gk)  subs(f,[x y],[xk - gk*gradx,yk - gk*grady])
% 
% GoldenSec(g,-5,5,0.01,0.01,200,0.618)

[xk,yk,k] = Steep(f,xk,yk,1,1)

%[xk,yk,k] = Steep(f,xk,yk,1,2)

%[xk,yk,k] = Steep(f,xk,yk,1,3)
[xxk,yyk,kk] = Newton(f,xk,yk,1)
[xxxk,yyyk,kkk] = LevMarq(f,xk,yk,1)
%% 
% Steepest Descent

function [xk,yk,k] = Steep(f,xk,yk,gk,mode)
%mode is about gk 1<-const 2<-minimizing f(x - gk*grad(f(x))) 3<-Armijo
k=0;
max_it = 100;
epsilon = 0.03;
syms x y gk
    
    
   while(k<max_it)
      df = gradient(f);
      dfx = df(1,1);
      dfy = df(2,1);
%     dfx = diff(f,x);
%     dfy = diff(f,y);
        

    gradx = double(subs(dfx,[x y],[xk yk]));
    grady = double(subs(dfy,[x y],[xk yk]));
    if(mode==1)
        gk=0.1;
    end
    if(mode==2)
        g = @(gk)  subs(f,[x y],[xk - gk*gradx,yk - gk*grady]);
        gk = GoldenSec(g,-5,5,0.01,0.01,200,0.618);
       

     
    end
    if(mode == 3)
        gk = Armijo(f,xk,yk,gradx,grady,-df,df,1,0.007,0.5);
    end

    


       if(sqrt(gradx^2+grady^2)<epsilon) %this might be better elsewhere
           break;
       end
    

        xk =double( xk - gk*gradx);
        yk = double(yk - gk*grady);
        
        k = k+1;
       kvec(k) = k;
       fvec(k) = double(subs(f,[x y],[xk yk]));

       plot(kvec,fvec);
       xlabel('Iterations')
       ylabel('F(xk,yk)')
       title('SteepestDescent')
   end

end

%% 
% Newton

function [xk,yk,k] = Newton(f,xk,yk,mode)
k=0;
max_iter = 100;
epsilon = 0.03;

syms x y gk;


while(k<max_iter)
df = gradient(f);
hess = -inv(hessian(f));

dfx = df(1,1);
dfy = df(2,1);
dk = hess * gradient(f);
tmpx = dk(1,1);
tmpy = dk(2,1);
gradx = double(subs(tmpx,[x y],[-1 1]));
grady = double(subs(tmpy,[x y],[-1 1])); 

if(sqrt(double(subs(dfx,[x y],[-1 1]))^2 + double(subs(dfy,[x y],[-1 1]))^2 )< epsilon)
    break;
end
if(mode==1)
    gk = 0.9;
end
if(mode==2)
     

    g = @(gk)  subs(f,[x y],[xk - gk*gradx,yk - gk*grady]);
        gk = GoldenSec(g,-5,5,0.01,0.01,200,0.618);
end
if(mode==3)
    gk = Armijo(f,xk,yk,gradx,grady,dk,df,0.05,0.09,0.1);
end





xk = xk - gk*gradx;
yk = yk - gk*grady;

k = k+1;

kvec(k) = k;
fvec(k) = double(subs(f,[x y],[xk yk]));

plot(kvec,fvec);
    xlabel('Iterations')
       ylabel('F(xk,yk)')
       title('Newton')

end
end
%% 
% Μέθοδος Levenberg-Marquardt

function [xk,yk,k] = LevMarq(f,xk,yk,mode)

figure;


syms x y;
epsilon = 0.03;
k=0;

flag=true;
hess = hessian(f);
df = gradient(f);
max_iter = 100;
m=0.1;
mv = m*[1 1];

hess = hess + diag(mv);
hess = double(subs(hess,[x y],[xk yk]));

while(double(norm(subs(df,[x y],[xk yk])))>epsilon || k<100)

  

hess = hessian(f);
hess = double(subs(hess,[x y],[xk yk]));
m=0.1;
mv = m*[1 1];
flag=true;
while(flag)
   
    try chol(hess);
        flag=false;
        
    catch
        flag=true;
        
        hess = hess+diag(mv);
        
    end
end

df = gradient(f);
df = double(subs(df,[x y],[xk yk]));

dk = inv(hess)*df;
gradx = double(df(1,1));
grady = double(df(2,1));

if(mode==1)
    gk=0.1;
end
if(mode==2)
    
    g = @(gk)  subs(f,[x y],[xk - gk*gradx,yk - gk*grady]);
    gk = GoldenSec(g,-5,5,0.01,0.01,200,0.618);
end
if(mode==3)
   gk =  Armijo(f,xk,yk,gradx,grady,-dk,df,5,0.001,0.2);
end

xk = xk - gk*dk(1,1);
yk = yk - gk*dk(2,1);
k = k+1;
kvec(k) = k;
fvec(k) = double(subs(f,[x y],[xk yk]));

plot(kvec,fvec);
xlabel('Iterations')
       ylabel('F(xk,yk)')
       title('Levenberg-Marquardt')

end

end
%% 
% 
% 
% 
% 
% Armijo

function [gk] = Armijo(f,xk,yk,gradx,grady,dk,df,s_start,alpha_start,beta_start)
syms x y ;
s = s_start;
alpha = alpha_start;
beta = beta_start;
m=1;
ffx = double(subs(f,[x y],[xk yk]));
dk = subs(dk,[x y],[xk yk]);
gk = s * beta^m;
tmpx = xk - gk*gradx;
tmpy = xk -gk*grady;
ffx_next = double(subs(f,[x y],[tmpx tmpy]));
tmmp = double(subs(df,[x y],[xk yk]));
tmmmp = double(ffx +alpha*beta^m*s *dk' * tmmp);
while( tmmmp > ffx_next)
    gk = s * beta^m;
    tmpx = xk - gk*gradx;
    tmpy = xk -gk*grady;
    ffx_next = double(subs(f,[x y],[tmpx tmpy]));
    m = m+1;
end


end
%% 
% Golden Section method from previous exercise

function [a,b,k] = GoldenSec(f,a,b,e,l,itter,gamma)
%uncommenting for (k,a) (k,b)

%itter = 

% figure;
% hold on;
    k=1;
    x1 = a + (1-gamma)*(b-a);
    x2 = a + gamma*(b-a);
    fx1 = f(x1);
    fx2 = f(x2);
    

    while (abs(b-a)>l) && (k<itter)
        
        
        if fx1 > fx2
            a = x1;
            
            x1 = x2;
            x2 = a + gamma*(b-a);

            
            fx1 = fx2;
            fx2 = f(x2);
            
            %plot(x1,fx1,'rx') %(x1,fx1)
             %plot(x2,fx2,'bo')
        else
            
            b = x2;
            x2 = x1;
            x1 = a + (1-gamma)*(b-a);
            
            
            fx2 = fx1;
            fx1 = f(x1);
            
            %plot(x1,fx1,'gx')
            %plot(x2,fx2,'go')

        end
%         plot(k,a,'rx')   %uncomment this one
%         plot(k,b,'bx')
        k = k+1;
        
    end
    k = k+1;
end
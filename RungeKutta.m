clear all;

Midpoint_method = zeros(1,10);
Midified_method = zeros(1,10);
Heun_method = zeros(1,10);
% N=10,h=0.2,ti=0.2i,w0=0.5
w0=0.5;
t=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0];

Midpoint_method(1,1) = w0;
Midified_method(1,1) = w0;
Heun_method(1,1) = w0;

for i = 1:10
    Midpoint_method(i+1) = 1.22*Midpoint_method(i) - 0.0088*i^2 - 0.008*i + 0.218;
    Midified_method(i+1) = 1.22*Midified_method(i) - 0.0088*i^2 - 0.008*i + 0.216;
    Heun_method(i+1) = 1.22*Heun_method(i) - 0.0088*i^2 - 0.008*i + 0.2173;
end
disp('-------------------------------------------------------------------')
disp('  ti      Midpoint_method      Midified_method        Heun_method')
disp('-------------------------------------------------------------------')
for i = 1:11
    fprintf('%2f         %10f         %10f         %10f\n',t(i),Midpoint_method(i),Midified_method(i),Heun_method(i))
end
disp('-------------------------------------------------------------------')

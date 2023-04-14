%    y' = y - t^2 + 1;   0 <= t <= 2;   y(0) = 0.5;
clear all;

w = zeros(10,1);
t = [0.0; 0.2; 0.4; 0.6; 0.8; 1.0; 1.2; 1.4; 1.6; 1.8; 2.0;];
a = 0;
b = 2;
N = 10;
% y(a) = alpha
alpha = 0.5;
h = (b-a)/N;
w(1) = alpha;
N = 10;

for i = 1:10
    w(i+1) = w(i) + h * (w(i) - (t(i))^2 + 1);
end

disp('---------------------------------------')
disp('    ti                  wi')
for i = 1:10
 fprintf('%10f       %10f\n',t(i),w(i))  
end
disp('---------------------------------------')

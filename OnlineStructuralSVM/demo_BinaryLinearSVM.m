% This is a demo about how to use structural svm to learn a standard linear SVM.
% run compile.m first to compile the c files

clear
clc
close all

% -------------------------------------------------------------------------
% Generate data
% -------------------------------------------------------------------------

th = pi/3 ;
c = cos(th) ;
s = sin(th) ;

patterns = {} ;
labels = {} ;
for i=1:100
    patterns{i} = diag([2 .5]) * randn(2, 1) ;
    labels{i}   = 2*(randn > 0) - 1 ;
    patterns{i}(2) = patterns{i}(2) + labels{i} ;
    patterns{i} = [c -s ; s c] * patterns{i}  ;
end

% -------------------------------------------------------------------------
% Run SVM struct
% -------------------------------------------------------------------------

% use global variable to communicate with @findMVC_BinaryLinearSVM without
% passing the data, which can be very expensive.
global patterns2train;
global labels2train  ;
patterns2train = patterns;
labels2train   = labels;
w_init = ones(2,1);
param.C = 1.0;
param.max_num_iterations = 500;
param.max_num_constraints =  10000;

w = trainOnlineStructSVM(numel(labels2train), w_init, @findMVC_BinaryLinearSVM, param);


% -------------------------------------------------------------------------
% Plots
% -------------------------------------------------------------------------

figure(1) ; clf ; hold on ;
x = [patterns2train{:}] ;
y = [labels2train{:}] ;
plot(x(1, y>0), x(2,y>0), 'b*') ;
plot(x(1, y<0), x(2,y<0), 'ro') ;
set(line([0 w(1)], [0 w(2)]), 'color', 'k', 'linewidth', 4) ;
set(line(10*[w(2) -w(2)], 10*[-w(1) w(1)]), 'color', 'k', 'linewidth', 2, 'linestyle', '-') ;
xlim([-3 3]) ;
ylim([-3 3]) ;
axis equal;
axis([-3 3 -3 3]);
w

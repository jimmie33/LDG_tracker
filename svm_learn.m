function w = svm_learn(feature_tables,cost_tables,w)
%% 
% labels are [1...1],corresponding to the first element in pattern

global patterns2train;
global costs2train ;
patterns2train = feature_tables;
costs2train = cost_tables;
% labels2train   = labels;
param.C = 1.0;
param.max_num_iterations = 500;
param.max_num_constraints =  10000;

w = trainOnlineStructSVM(numel(costs2train), w, @findMVC_BinaryLinearSVM, param);

end


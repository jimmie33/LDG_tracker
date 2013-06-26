function idx = svmTrackerDo (tracker,sample)
switch tracker.solver
    case 0
%         [~,resp] = svmclassify_my (tracker.clsf,sample);
        [~,idx] = min(sample*tracker.clsf.w');
    case 1
        [~,idx] = max(sample*tracker.clsf.w');
    case 2
        [~,idx] = max(sample*tracker.clsf.w');
    case 4 % psvm
        [~,idx] = max(sample*tracker.clsf.w);
    case 5 % tvm
        [~,idx] = min(sample*tracker.clsf.w');
end
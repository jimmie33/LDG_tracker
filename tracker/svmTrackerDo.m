function svmTrackerDo (sample)
global sampler
global svm_tracker
switch svm_tracker.solver
    case 0
%         [~,resp] = svmclassify_my (svm_tracker.clsf,sample);
        [~,idx] = min(sample*svm_tracker.clsf.w');
        svm_tracker.output = sampler.state_dt(idx,:);
    case 1
        [~,idx] = max(sample*svm_tracker.clsf.w');
        svm_tracker.output = sampler.state_dt(idx,:);
    case 2
        [~,idx] = max(sample*svm_tracker.clsf.w');
        svm_tracker.output = sampler.state_dt(idx,:);
    case 4 % psvm
        [~,idx] = max(sample*svm_tracker.clsf.w);
        svm_tracker.output = sampler.state_dt(idx,:);
    case 5 % tvm
        [~,idx] = min(sample*svm_tracker.w');
        svm_tracker.output = sampler.state_dt(idx,:);
        svm_tracker.confidence = -(sample(idx,:)*svm_tracker.w'+svm_tracker.Bias);
        
end
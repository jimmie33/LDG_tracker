function updateTrackerExperts
global svm_tracker

if numel(svm_tracker.experts) < svm_tracker.max_expert_sz
    svm_tracker.experts{end+1} = svm_tracker.experts{end};
else
    svm_tracker.experts(1:end-1) = svm_tracker.experts(2:end);
end
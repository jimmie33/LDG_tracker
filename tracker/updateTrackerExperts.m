function updateTrackerExperts
global config
global svm_tracker
global experts

if numel(experts) < config.max_expert_sz
    svm_tracker.update_count = 0;
    experts{end}.snapshot = svm_tracker;
    experts{end+1} = experts{end};
else
    svm_tracker.update_count = 0;
    experts{end}.snapshot = svm_tracker;
    experts(1:end-1) = experts(2:end);
end
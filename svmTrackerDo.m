function idx = svmTrackerDo (tracker,sample)
[~,resp] = svmclassify_my (tracker.clsf,sample);
[~,idx] = min(resp);
function tracker = initSvmTracker (tracker,sample,label)
% tracker.feat_w = feat_w;
tracker.clsf = svmtrain( sample, label,'kernel_function','rbf',...
    'rbf_sigma',tracker.sigma,'boxconstraint',tracker.C,'autoscale','false');
tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
if size(tracker.clsf.Alpha,1) > tracker.sv_size
    [~, Idx]= sort(abs(tracker.clsf.Alpha),1,'descend');
    tracker.clsf.Alpha = tracker.clsf.Alpha(Idx(1:tracker.sv_size),1);
    tracker.clsf.SupportVectors = tracker.clsf.SupportVectors(Idx(1:tracker.sv_size),:);
    tracker.sv_full = tracker.sv_full(Idx(1:tracker.sv_size),:);
    tracker.sv_label = label(Idx(1:tracker.sv_size),:);
end
tracker.clsf.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
function tracker = updateSvmTracker(tracker,sample,label)
global config
% tracker.feat_w = feat_w;
sample = [sample; tracker.sv_full];
label = [label; tracker.sv_label];% positive:1 negative:0
tracker.clsf = svmtrain( sample, label,'boxconstraint',tracker.C,'autoscale','false');
tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
if config.verbose, fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(tracker.clsf.Alpha,1)); end
if size(tracker.clsf.Alpha,1) > tracker.sv_size
    [~, Idx]= sort(abs(tracker.clsf.Alpha),1,'descend');
    tracker.clsf.Alpha = tracker.clsf.Alpha(Idx(1:tracker.sv_size),1);
    tracker.clsf.SupportVectors = tracker.clsf.SupportVectors(Idx(1:tracker.sv_size),:);
    tracker.sv_full = tracker.sv_full(Idx(1:tracker.sv_size),:);
    tracker.sv_label = label(Idx(1:tracker.sv_size),:);
end
tracker.clsf.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
function tracker = initSvmTracker (tracker,sample,label)
% tracker.feat_w = feat_w;
global config

switch tracker.solver
    case 0 % matlab built-in
        tracker.clsf = svmtrain( sample, label,'kernel_function','rbf',...
            'rbf_sigma',tracker.sigma,'boxconstraint',tracker.C,'autoscale','false');
        tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
        tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
        if config.verbose 
            fprintf('matlab: feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(tracker.clsf.Alpha,1)); 
        end
        tracker.clsf.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
        if size(tracker.clsf.Alpha,1) > tracker.sv_size
            delta_w = dot(tracker.clsf.SupportVectors,tracker.clsf.SupportVectors,2).*...
                (tracker.clsf.Alpha.^2);
            [~, Idx]= sort(delta_w,1,'descend');
            tracker.clsf.Alpha = tracker.clsf.Alpha(Idx(1:tracker.sv_size),1);
            tracker.clsf.SupportVectors = tracker.clsf.SupportVectors(Idx(1:tracker.sv_size),:);
            tracker.sv_full = tracker.sv_full(Idx(1:tracker.sv_size),:);
            tracker.sv_label = label(Idx(1:tracker.sv_size),:);
        end
        
    case 1 % liblinear without storing svs
        label = double(label);
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        if config.verbose 
            fprintf('liblinear: feat_d: %d; train_num: %d\n',size(sample,2),size(sample,1)); 
        end
        tracker.clsf = trainll(label, sparse(sample),['-B 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
        tracker.clsf.w = tracker.clsf.w(1:end-1);
    case 2 % libsvm
        label = double(label);
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        
        tracker.clsf = svmlibtrain(double(label),sample,['-b 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
        tracker.sv_full = sample(tracker.clsf.sv_indices,:);
        tracker.sv_label = label(tracker.clsf.sv_indices,:);
        if config.verbose
            fprintf('libsvm: feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),tracker.clsf.totalSV); 
        end
        if tracker.clsf.totalSV > tracker.sv_size
            [~, Idx]= sort(abs(tracker.clsf.sv_coef),1,'descend');
            tracker.clsf.sv_coef = tracker.clsf.sv_coef(Idx(1:tracker.sv_size),1);
            tracker.clsf.SVs = tracker.clsf.SVs(Idx(1:tracker.sv_size),:);
            tracker.sv_full = tracker.sv_full(Idx(1:tracker.sv_size),:);
            tracker.sv_label = label(Idx(1:tracker.sv_size),:);
        end
        tracker.clsf.w = tracker.clsf.sv_coef'*tracker.clsf.SVs(:,1:end);
end
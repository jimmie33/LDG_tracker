function tracker = initSvmTracker (tracker,sample,label)
% tracker.feat_w = feat_w;
global config

switch tracker.solver
    case 0 % matlab built-in
        tracker.clsf = svmtrain( sample, label,'boxconstraint',tracker.C,'autoscale','false');
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
        tracker.clsf = trainll((double(label)-0.5)*2, sparse(sample),['-s 3 -B 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
        score = 2*(label-0.5).*([sample,ones(size(sample,1),1)]*tracker.clsf.w');
        sv_mask = (score <= 1.3 & score > 0.7);
        tracker.sv_full = sample(sv_mask,:);
        tracker.sv_label = label(sv_mask);
        
        tracker.clsf.w = tracker.clsf.w(1:end-1);
    case 2 % libsvm
        label = double(label);
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        
        tracker.clsf = svmlibtrain((double(label)-0.5)*2,sample,['-t 0 -b 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
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
        
    case 4 % psvm
        [tracker.clsf.w, tracker.clsf.gamma, y] = psvm_my(sample,2*(label-0.5),0,-1,0,1);

        mask = abs(y) <0.05 | abs(y)>0.1;
        tracker.sv_full = sample(mask,:);
        tracker.sv_label = label(mask,:);
        
        if config.verbose
            fprintf('psvm: feat_d: %d; train_num: %d; svs: %d \n',size(sample,2),size(sample,1),size(tracker.sv_full,1)); 
        end
        
    case 5 % tvm
        tracker.clsf = svmtrain( sample, label,'boxconstraint',tracker.C,'autoscale','false');
        tracker.clsf.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
        tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
        tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
        
        tracker.pos_sv = tracker.sv_full(tracker.sv_label>0.5,:);
        tracker.pos_w = ones(size(tracker.pos_sv,1),1);
        tracker.neg_sv = tracker.sv_full(tracker.sv_label<0.5,:);
        tracker.neg_w = ones(size(tracker.neg_sv,1),1);
        
        % calculate distance matrix
        tracker.pos_dis = squareform(pdist(tracker.pos_sv));
        tracker.neg_dis = squareform(pdist(tracker.neg_sv)); 
        
end
function tracker = updateSvmTracker(tracker,sample,label)
global config
% tracker.feat_w = feat_w;

switch tracker.solver
    case 0 % matlab built-in
        sample = [tracker.sv_full; sample];
        label = [tracker.sv_label;label];% positive:1 negative:0
%         n_pos = sum(label>0.5);
%         n_neg = size(label,1) - n_pos;
%         n_pos_sv = sum(tracker.sv_label>0.5);
%         n_neg_sv = size(tracker.sv_label,1) - n_pos_sv;
%         c_pos_sv = (n_pos_sv+n_neg_sv)/(2*n_pos_sv);
%         c_neg_sv = (n_pos_sv+n_neg_sv)/(2*n_neg_sv);
%         c_pos = (n_pos+n_neg)/(2*n_pos);
%         c_neg = (n_pos+n_neg)/(2*n_neg);
%         C = zeros(size(label));
%         C(label>0.5) = c_pos;
%         C(label<0.5) = c_neg;
%         C(find(tracker.sv_label>0.5)) = 5*c_pos_sv;
%         C(find(tracker.sv_label<0.5)) = 5*c_neg_sv;
%         C = C/max(C);
        
        if config.verbose
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',tracker.C,...
                'autoscale','false','options',statset('Display','final','MaxIter',5000));
        else
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',tracker.C,...
                'autoscale','false','options',statset('MaxIter',5000));
        end
        tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
        tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
        if config.verbose
            fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(tracker.clsf.Alpha,1)); 
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
%         fprintf('--------------------- w norm: %f\n',norm(tracker.clsf.w));
%         if min(-2*(tracker.sv_label-0.5).*...
%                 ((tracker.clsf.SupportVectors*tracker.clsf.w') + tracker.clsf.Bias))<0
%             keyboard
%         end
        
    case 1 % liblinear: smoothing without storing svs
        label = double(label);
        sample = [tracker.sv_full; sample];
        label = [tracker.sv_label;label];% positive:1 negative:0
        
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(neg_num/(pos_num+neg_num));
        c_neg = num2str(pos_num/(pos_num+neg_num));
        if config.verbose 
            fprintf('feat_d: %d; train_num: %d\n',size(sample,2),size(sample,1)); 
        end
        w_old = tracker.clsf.w/norm(tracker.clsf.w);
        tracker.clsf = trainll((double(label)-0.5)*2, sparse(sample),['-s 3 -q -B 1',' -w1 ',c_pos,' -w0 ',c_neg]);
        
        % look for support vectors
        score = 2*(label-0.5).*([sample,ones(size(sample,1),1)]*tracker.clsf.w');
        sv_mask = (score <= 1.3 & score > 0.7);
        tracker.sv_full = sample(sv_mask,:);
        tracker.sv_label = label(sv_mask);
        
        tracker.clsf.w = w_old*0.0 + (tracker.clsf.w(1:end-1)/norm(tracker.clsf.w(1:end-1)))*1;
        
    case 2
        sample = [sample; tracker.sv_full];
        label = [double(label); tracker.sv_label];% positive:1 negative:0
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        
        tracker.clsf = svmlibtrain((double(label)-0.5)*2,sample,['-t 0 -q -b 1',' -w1 ',c_pos,' -w0 ',c_neg]);
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
    otherwise
        
end
function initSvmTracker (sample,label,fuzzy_weight)
% svm_tracker.feat_w = feat_w;
global config
global svm_tracker;
global sampler;

switch svm_tracker.solver
    case 0 % matlab built-in
        svm_tracker.clsf = svmtrain( sample, label,'boxconstraint',svm_tracker.C,'autoscale','false');
        svm_tracker.sv_full = sample(svm_tracker.clsf.SupportVectorIndices,:);
        svm_tracker.sv_label = label(svm_tracker.clsf.SupportVectorIndices,:);
        if config.verbose 
            fprintf('matlab: feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(svm_tracker.clsf.Alpha,1)); 
        end
        svm_tracker.clsf.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors;
        if size(svm_tracker.clsf.Alpha,1) > svm_tracker.sv_size
            delta_w = dot(svm_tracker.clsf.SupportVectors,svm_tracker.clsf.SupportVectors,2).*...
                (svm_tracker.clsf.Alpha.^2);
            [~, Idx]= sort(delta_w,1,'descend');
            svm_tracker.clsf.Alpha = svm_tracker.clsf.Alpha(Idx(1:svm_tracker.sv_size),1);
            svm_tracker.clsf.SupportVectors = svm_tracker.clsf.SupportVectors(Idx(1:svm_tracker.sv_size),:);
            svm_tracker.sv_full = svm_tracker.sv_full(Idx(1:svm_tracker.sv_size),:);
            svm_tracker.sv_label = label(Idx(1:svm_tracker.sv_size),:);
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
        svm_tracker.clsf = trainll((double(label)-0.5)*2, sparse(sample),['-s 3 -B 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
        score = 2*(label-0.5).*([sample,ones(size(sample,1),1)]*svm_tracker.clsf.w');
        sv_mask = (score <= 1.3 & score > 0.7);
        svm_tracker.sv_full = sample(sv_mask,:);
        svm_tracker.sv_label = label(sv_mask);
        
        svm_tracker.clsf.w = svm_tracker.clsf.w(1:end-1);
    case 2 % libsvm
        label = double(label);
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        
        svm_tracker.clsf = svmlibtrain((double(label)-0.5)*2,sample,['-t 0 -b 1 -q',' -w1 ',c_pos,' -w0 ',c_neg]);
        svm_tracker.sv_full = sample(svm_tracker.clsf.sv_indices,:);
        svm_tracker.sv_label = label(svm_tracker.clsf.sv_indices,:);
        if config.verbose
            fprintf('libsvm: feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),svm_tracker.clsf.totalSV); 
        end
        if svm_tracker.clsf.totalSV > svm_tracker.sv_size
            [~, Idx]= sort(abs(svm_tracker.clsf.sv_coef),1,'descend');
            svm_tracker.clsf.sv_coef = svm_tracker.clsf.sv_coef(Idx(1:svm_tracker.sv_size),1);
            svm_tracker.clsf.SVs = svm_tracker.clsf.SVs(Idx(1:svm_tracker.sv_size),:);
            svm_tracker.sv_full = svm_tracker.sv_full(Idx(1:svm_tracker.sv_size),:);
            svm_tracker.sv_label = label(Idx(1:svm_tracker.sv_size),:);
        end
        svm_tracker.clsf.w = svm_tracker.clsf.sv_coef'*svm_tracker.clsf.SVs(:,1:end);
        
    case 4 % psvm
        [svm_tracker.clsf.w, svm_tracker.clsf.gamma, y] = psvm_my(sample,2*(label-0.5),0,-1,0,1);

        mask = abs(y) <0.05 | abs(y)>0.1;
        svm_tracker.sv_full = sample(mask,:);
        svm_tracker.sv_label = label(mask,:);
        
        if config.verbose
            fprintf('psvm: feat_d: %d; train_num: %d; svs: %d \n',size(sample,2),size(sample,1),size(svm_tracker.sv_full,1)); 
        end
        
    case 5 % tvm
        
        num_newsample = size(sample,1);

%         sample = [svm_tracker.pos_sv;svm_tracker.neg_sv; sample];
%         label = [ones(size(svm_tracker.pos_sv,1),1);zeros(size(svm_tracker.neg_sv,1),1);label];% positive:1 negative:0
        sample_w = fuzzy_weight;
       
        pos_mask = label>0.5;
        neg_mask = ~pos_mask;
        s1 = sum(sample_w(pos_mask));
        s2 = sum(sample_w(neg_mask));
        
        sample_w(pos_mask) = sample_w(pos_mask)*s2;
        sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
        C = max(svm_tracker.C*sample_w/sum(sample_w),0.001);
        
        svm_tracker.clsf = svmtrain( sample, label,'boxconstraint',C,'autoscale','false');
        
        svm_tracker.struct_mat = eye(size(sample,2));
       
        svm_tracker.clsf.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors;
        svm_tracker.w = svm_tracker.clsf.w;
        svm_tracker.Bias = svm_tracker.clsf.Bias;
        svm_tracker.sv_label = label(svm_tracker.clsf.SupportVectorIndices,:);
        svm_tracker.sv_full = sample(svm_tracker.clsf.SupportVectorIndices,:);
        
        svm_tracker.pos_sv = svm_tracker.sv_full(svm_tracker.sv_label>0.5,:);
        svm_tracker.pos_w = ones(size(svm_tracker.pos_sv,1),1);
        svm_tracker.neg_sv = svm_tracker.sv_full(svm_tracker.sv_label<0.5,:);
        svm_tracker.neg_w = ones(size(svm_tracker.neg_sv,1),1);
        
        % compute real margin
        pos2plane = -svm_tracker.pos_sv*svm_tracker.w';
        neg2plane = -svm_tracker.neg_sv*svm_tracker.w';
        svm_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(svm_tracker.w);
        
        % calculate distance matrix
        if size(svm_tracker.pos_sv,1)>1
            svm_tracker.pos_dis = squareform(pdist(svm_tracker.pos_sv));
        else
            svm_tracker.pos_dis = inf;
        end
        svm_tracker.neg_dis = squareform(pdist(svm_tracker.neg_sv)); 
        
        %% intialize tracker experts
        svm_tracker.experts{1}.w = svm_tracker.w;
        svm_tracker.experts{1}.Bias = svm_tracker.Bias;
        svm_tracker.experts{1}.score = [];
        
        svm_tracker.experts{2} = svm_tracker.experts{1};
        svm_tracker.experts{3} = svm_tracker.experts{1};
        
        svm_tracker.template = sampler.template;
        
%         svm_tracker.experts{2}.w = svm_tracker.w;
%         svm_tracker.experts{2}.Bias = svm_tracker.Bias;
%         svm_tracker.experts{2}.score = [];
%         
%         svm_tracker.experts{3}.w = svm_tracker.w;
%         svm_tracker.experts{3}.Bias = svm_tracker.Bias;
%         svm_tracker.experts{3}.score = [];
        
        % structral information
%         svm_tracker.pos_corr = zeros(size(svm_tracker.pos_sv,2),size(svm_tracker.pos_sv,2),...
%             size(svm_tracker.pos_sv,1));
%         for k = 1:size(svm_tracker.pos_sv,1)
%             svm_tracker.pos_corr(:,:,k) = svm_tracker.pos_sv(k,:)'*svm_tracker.pos_sv(k,:);
%         end
%         svm_tracker.pos_ms = svm_tracker.pos_corr;
        
end
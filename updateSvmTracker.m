function tracker = updateSvmTracker(tracker,sample,label)
global config
% tracker.feat_w = feat_w;

switch tracker.solver
    case 0 % matlab built-in
        % test
        score = -(sample*tracker.clsf.w'+tracker.clsf.Bias);
        pos_score = score(label>0.5);
        neg_score = score(label<0.5);
        figure(4)
        bar([hist(pos_score,-1:0.25:1)',hist(neg_score,-1:0.25:1)']);
        
        mask = rand(size(label))>0.5 & (score > max(neg_score) | score<min(pos_score));
        sample = [tracker.sv_full; sample(mask,:)];
        label = [tracker.sv_label;label(mask)];% positive:1 negative:0
        
        
        if config.verbose
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',tracker.C/size(sample,1),...
                'autoscale','false','options',statset('Display','final','MaxIter',5000));
            fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(tracker.clsf.Alpha,1)); 
        else
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',tracker.C/size(sample,1),...
                'autoscale','false','options',statset('MaxIter',5000));
        end
        tracker.sv_full = sample(tracker.clsf.SupportVectorIndices,:);
        tracker.sv_label = label(tracker.clsf.SupportVectorIndices,:);
        
%         tracker.sv_full = tracker.sv_full(tracker.sv_label>0.5,:);
%         tracker.sv_label = tracker.sv_label(tracker.sv_label>0.5);
%         
        tracker.clsf.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
        if size(tracker.clsf.Alpha,1) > tracker.sv_size
%             delta_w = dot(tracker.clsf.SupportVectors,tracker.clsf.SupportVectors,2).*...
%                 (tracker.clsf.Alpha.^2);
%             [~, Idx]= sort(delta_w,1,'descend');
            score_sv = -(tracker.sv_full*tracker.clsf.w'+tracker.clsf.Bias);
            [~, Idx]= sort(abs(score_sv),1,'ascend');
%             tracker.clsf.Alpha = tracker.clsf.Alpha(abs(score_sv)<2,1);
%             tracker.clsf.SupportVectors = tracker.clsf.SupportVectors(Idx(1:tracker.sv_size),:);
%             if(sum(1- (abs(score_sv)<2)) > 0)
%                 keyboard
%             end
            tracker.sv_full = tracker.sv_full(abs(score_sv)<2 | tracker.sv_label>0.5,:);
            tracker.sv_label = tracker.sv_label(abs(score_sv)<2 | tracker.sv_label>0.5,:);
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
            tracker.sv_label = tracker.sv_label(Idx(1:tracker.sv_size),:);
        end
        tracker.clsf.w = tracker.clsf.sv_coef'*tracker.clsf.SVs(:,1:end);
    case 4 % psvm
        sample = [sample; tracker.sv_full];
        label = [double(label); tracker.sv_label];% positive:1 negative:0
        [tracker.clsf.w, tracker.clsf.gamma] = psvm(sample,2*(label-0.5),0,-1,0,1);
%         y = abs(y);
        mask_pos = label>0.5 & rand(size(label))>0.0;
        mask_neg = label<0.5 & rand(size(label))<0.3;
        if size(tracker.sv_full,1)<500
            tracker.sv_full = sample(mask_pos | mask_neg,:);
            tracker.sv_label = label(mask_pos | mask_neg,:);
        end
        if config.verbose
            fprintf('psvm: feat_d: %d; train_num: %d; svs: %d \n',size(sample,2),size(sample,1),size(tracker.sv_full,1)); 
        end
    case 5 % tvm
%         figure(4)
%         bar([size(tracker.pos_sv,1),size(tracker.neg_sv,1)]);
        
        score = -(sample*tracker.clsf.w'+tracker.clsf.Bias);
        pos_score = score(label>0.5);
        neg_score = score(label<0.5);
        mask = (rand(size(label))>0.5); %& ...
            %(score > max(neg_score) | score<min(pos_score));
        sample = sample(mask,:);
        label = label(mask,:);
        num_newsample = size(sample,1);

        sample = [tracker.pos_sv;tracker.neg_sv; sample];
        label = [ones(size(tracker.pos_sv,1),1);zeros(size(tracker.neg_sv,1),1);label];% positive:1 negative:0
        sample_w = [tracker.pos_w;tracker.neg_w;ones(num_newsample,1)];
       
        pos_mask = label>0.5;
        neg_mask = ~pos_mask;
        s1 = sum(sample_w(pos_mask));
        s2 = sum(sample_w(neg_mask));
        
        sample_w(pos_mask) = sample_w(pos_mask)*s2;
        sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
        C = tracker.C*sample_w/sum(sample_w);
        
        % whitening ********************
%         tmp_msk = tracker.pos_w > 1;
%         sigma = sum(tracker.pos_corr(:,:,tmp_msk),3)-sum(tracker.pos_ms(:,:,tmp_msk),3);
%         wht = speye(size(tracker.pos_sv,2))+tracker.lambda*sigma;
%         tracker.whitening = sqrtm(inv(wht));
        
        if config.verbose
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',C,...
                'autoscale','false','options',statset('Display','final','MaxIter',5000));
            fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(tracker.clsf.Alpha,1)); 
        else
            tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',C,...
                'autoscale','false','options',statset('MaxIter',5000));
        end
        %**************************
        if ~isempty(tracker.w)
            tracker.w = 0.9*tracker.w + 0.1*tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
            tracker.Bias = 0.9*tracker.Bias + 0.1*tracker.clsf.Bias;
        else
            tracker.w = tracker.clsf.Alpha'*tracker.clsf.SupportVectors;
            tracker.Bias = tracker.clsf.Bias;
        end
        tracker.clsf.w = tracker.w;
        % get the idx of new svs
        sv_idx = tracker.clsf.SupportVectorIndices;
        sv_old_sz = size(tracker.pos_sv,1)+size(tracker.neg_sv,1);
        sv_new_idx = sv_idx(sv_idx>sv_old_sz);
        sv_new = sample(sv_new_idx,:);
        sv_new_label = label(sv_new_idx,:);
        
        num_sv_pos_new = sum(sv_new_label);
        
        % update pos_dis, pos_w and pos_sv
        pos_sv_new = sv_new(sv_new_label>0.5,:);
        if ~isempty(pos_sv_new)
            if size(pos_sv_new,1)>1
                pos_dis_new = squareform(pdist(pos_sv_new));
            else
                pos_dis_new = 0;
            end
            pos_dis_cro = pdist2(tracker.pos_sv,pos_sv_new);
            tracker.pos_dis = [tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
            tracker.pos_sv = [tracker.pos_sv;pos_sv_new];
            tracker.pos_w = [tracker.pos_w;ones(num_sv_pos_new,1)];
            
            % update structrual information ********************
%             pos_corr = zeros(size(tracker.pos_sv,2),size(tracker.pos_sv,2),...
%                 num_sv_pos_new);
%             for k = 1:num_sv_pos_new
%                 pos_corr(:,:,k) = pos_sv_new(k,:)'*pos_sv_new(k,:);
%             end
%             tracker.pos_corr = cat(3,tracker.pos_corr,pos_corr);
%             tracker.pos_ms = cat(3,tracker.pos_ms,pos_corr);
        end
        
        % update neg_dis, neg_w and neg_sv
        neg_sv_new = sv_new(sv_new_label<0.5,:);
        if ~isempty(neg_sv_new)
            if size(neg_sv_new,1)>1
                neg_dis_new = squareform(pdist(neg_sv_new));
            else
                neg_dis_new = 0;
            end
            neg_dis_cro = pdist2(tracker.neg_sv,neg_sv_new);
            tracker.neg_dis = [tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
            tracker.neg_sv = [tracker.neg_sv;neg_sv_new];
            tracker.neg_w = [tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
        end
        
        tracker.pos_dis = tracker.pos_dis + diag(inf*ones(size(tracker.pos_dis,1),1));
        tracker.neg_dis = tracker.neg_dis + diag(inf*ones(size(tracker.neg_dis,1),1));
        
        % shrink svs
        % check if to remove
        if size(tracker.pos_sv,1)+size(tracker.neg_sv,1)>tracker.B
            pos_score_sv = -(tracker.pos_sv*tracker.w'+tracker.Bias);
            neg_score_sv = -(tracker.neg_sv*tracker.w'+tracker.Bias);
            m_pos = abs(pos_score_sv) < tracker.m2;
            m_neg = abs(neg_score_sv) < tracker.m2;
            
            if config.verbose
                fprintf('remove svs: pos %d, neg %d \n',sum(~m_pos),sum(~m_neg));
            end
            if sum(m_pos) > 0
                tracker.pos_sv = tracker.pos_sv(m_pos,:);
                tracker.pos_w = tracker.pos_w(m_pos,:);
                tracker.pos_dis = tracker.pos_dis(m_pos,m_pos);
            end
            % update structrual information *******************
%             tracker.pos_corr = tracker.pos_corr(:,:,m_pos);
%             tracker.pos_ms = tracker.pos_ms(:,:,m_pos);
            if sum(m_neg)>0
                tracker.neg_sv = tracker.neg_sv(m_neg,:);
                tracker.neg_w = tracker.neg_w(m_neg,:);
                tracker.neg_dis = tracker.neg_dis(m_neg,m_neg);
            end
        end
        
        % check if to merge
        while size(tracker.pos_sv,1)+size(tracker.neg_sv,1)>tracker.B
            [mm_pos,idx_pos] = min(tracker.pos_dis(:));
            [mm_neg,idx_neg] = min(tracker.neg_dis(:));
            
            if mm_pos > mm_neg || size(tracker.pos_sv,1) <= tracker.B_p% merge negative samples
                if config.verbose
                    fprintf('merge negative samples: %d \n', size(tracker.neg_w,1))
                end
                
                [i,j] = ind2sub(size(tracker.neg_dis),idx_neg);
                w_i= tracker.neg_w(i);
                w_j= tracker.neg_w(j);
                merge_sample = (w_i*tracker.neg_sv(i,:)+w_j*tracker.neg_sv(j,:))/(w_i+w_j);                
                
                tracker.neg_sv([i,j],:) = []; tracker.neg_sv(end+1,:) = merge_sample;
                tracker.neg_w([i,j]) = []; tracker.neg_w(end+1,1) = w_i + w_j;
                
                tracker.neg_dis([i,j],:)=[]; tracker.neg_dis(:,[i,j])=[];
                neg_dis_cro = pdist2(tracker.neg_sv(1:end-1,:),merge_sample);
                tracker.neg_dis = [tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
            else
                if config.verbose
                    fprintf('merge positive samples: %d \n', size(tracker.pos_w,1))
                end
%                 if size(tracker.pos_w,1)<4
%                     keyboard
%                 end
                [i,j] = ind2sub(size(tracker.pos_dis),idx_pos);
                w_i= tracker.pos_w(i);
                w_j= tracker.pos_w(j);
                merge_sample = (w_i*tracker.pos_sv(i,:)+w_j*tracker.pos_sv(j,:))/(w_i+w_j);                
                
                % update structrual information *******
%                 pos_corr = (w_i/(w_i+w_j))*tracker.pos_corr(:,:,i) + ...
%                     (w_j/(w_i+w_j))*tracker.pos_corr(:,:,j);
%                 tracker.pos_corr(:,:,[i,j]) = [];
%                 tracker.pos_corr(:,:,end+1) = pos_corr;
%                 tracker.pos_ms(:,:,[i,j]) = [];
%                 tracker.pos_ms(:,:,end+1) = merge_sample'*merge_sample;
                
                
                tracker.pos_sv([i,j],:) = []; tracker.pos_sv(end+1,:) = merge_sample;
                tracker.pos_w([i,j]) = []; tracker.pos_w(end+1,1) = w_i + w_j;
                
                tracker.pos_dis([i,j],:)=[]; tracker.pos_dis(:,[i,j])=[];
                pos_dis_cro = pdist2(tracker.pos_sv(1:end-1,:),merge_sample);
                tracker.pos_dis = [tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
                
                
            end
            
        end
        
        
        
end
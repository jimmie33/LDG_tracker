function updateSvmTracker(sample,label,fuzzy_weight)
global config;
global svm_tracker;
% svm_tracker.feat_w = feat_w;

switch svm_tracker.solver
    case 0 % matlab built-in
        % test
        score = -(sample*svm_tracker.clsf.w'+svm_tracker.clsf.Bias);
        pos_score = score(label>0.5);
        neg_score = score(label<0.5);
        figure(4)
        bar([hist(pos_score,-1:0.25:1)',hist(neg_score,-1:0.25:1)']);
        
        mask = rand(size(label))>0.5 & (score > max(neg_score) | score<min(pos_score));
        sample = [svm_tracker.sv_full; sample(mask,:)];
        label = [svm_tracker.sv_label;label(mask)];% positive:1 negative:0
        
        
        if config.verbose
            svm_tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',svm_tracker.C/size(sample,1),...
                'autoscale','false','options',statset('Display','final','MaxIter',5000));
            fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(svm_tracker.clsf.Alpha,1)); 
        else
            svm_tracker.clsf = svmtrain_my( sample, label, 'boxconstraint',svm_tracker.C/size(sample,1),...
                'autoscale','false','options',statset('MaxIter',5000));
        end
        svm_tracker.sv_full = sample(svm_tracker.clsf.SupportVectorIndices,:);
        svm_tracker.sv_label = label(svm_tracker.clsf.SupportVectorIndices,:);
        
%         svm_tracker.sv_full = svm_tracker.sv_full(svm_tracker.sv_label>0.5,:);
%         svm_tracker.sv_label = svm_tracker.sv_label(svm_tracker.sv_label>0.5);
%         
        svm_tracker.clsf.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors;
        if size(svm_tracker.clsf.Alpha,1) > svm_tracker.sv_size
%             delta_w = dot(svm_tracker.clsf.SupportVectors,svm_tracker.clsf.SupportVectors,2).*...
%                 (svm_tracker.clsf.Alpha.^2);
%             [~, Idx]= sort(delta_w,1,'descend');
            score_sv = -(svm_tracker.sv_full*svm_tracker.clsf.w'+svm_tracker.clsf.Bias);
            [~, Idx]= sort(abs(score_sv),1,'ascend');
%             svm_tracker.clsf.Alpha = svm_tracker.clsf.Alpha(abs(score_sv)<2,1);
%             svm_tracker.clsf.SupportVectors = svm_tracker.clsf.SupportVectors(Idx(1:svm_tracker.sv_size),:);
%             if(sum(1- (abs(score_sv)<2)) > 0)
%                 keyboard
%             end
            svm_tracker.sv_full = svm_tracker.sv_full(abs(score_sv)<2 | svm_tracker.sv_label>0.5,:);
            svm_tracker.sv_label = svm_tracker.sv_label(abs(score_sv)<2 | svm_tracker.sv_label>0.5,:);
        end
%         fprintf('--------------------- w norm: %f\n',norm(svm_tracker.clsf.w));
%         if min(-2*(svm_tracker.sv_label-0.5).*...
%                 ((svm_tracker.clsf.SupportVectors*svm_tracker.clsf.w') + svm_tracker.clsf.Bias))<0
%             keyboard
%         end
        
    case 1 % liblinear: smoothing without storing svs
        label = double(label);
        sample = [svm_tracker.sv_full; sample];
        label = [svm_tracker.sv_label;label];% positive:1 negative:0
        
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(neg_num/(pos_num+neg_num));
        c_neg = num2str(pos_num/(pos_num+neg_num));
        if config.verbose 
            fprintf('feat_d: %d; train_num: %d\n',size(sample,2),size(sample,1)); 
        end
        w_old = svm_tracker.clsf.w/norm(svm_tracker.clsf.w);
        svm_tracker.clsf = trainll((double(label)-0.5)*2, sparse(sample),['-s 3 -q -B 1',' -w1 ',c_pos,' -w0 ',c_neg]);
        
        % look for support vectors
        score = 2*(label-0.5).*([sample,ones(size(sample,1),1)]*svm_tracker.clsf.w');
        sv_mask = (score <= 1.3 & score > 0.7);
        svm_tracker.sv_full = sample(sv_mask,:);
        svm_tracker.sv_label = label(sv_mask);
        
        svm_tracker.clsf.w = w_old*0.0 + (svm_tracker.clsf.w(1:end-1)/norm(svm_tracker.clsf.w(1:end-1)))*1;
        
    case 2
        sample = [sample; svm_tracker.sv_full];
        label = [double(label); svm_tracker.sv_label];% positive:1 negative:0
        pos_num = sum(label>0.5);
        neg_num = sum(label<0.5);
        c_pos = num2str(0.5*(pos_num+neg_num)/pos_num);
        c_neg = num2str(0.5*(pos_num+neg_num)/neg_num);
        
        svm_tracker.clsf = svmlibtrain((double(label)-0.5)*2,sample,['-t 0 -q -b 1',' -w1 ',c_pos,' -w0 ',c_neg]);
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
            svm_tracker.sv_label = svm_tracker.sv_label(Idx(1:svm_tracker.sv_size),:);
        end
        svm_tracker.clsf.w = svm_tracker.clsf.sv_coef'*svm_tracker.clsf.SVs(:,1:end);
    case 4 % psvm
        sample = [sample; svm_tracker.sv_full];
        label = [double(label); svm_tracker.sv_label];% positive:1 negative:0
        [svm_tracker.clsf.w, svm_tracker.clsf.gamma] = psvm(sample,2*(label-0.5),0,-1,0,1);
%         y = abs(y);
        mask_pos = label>0.5 & rand(size(label))>0.0;
        mask_neg = label<0.5 & rand(size(label))<0.3;
        if size(svm_tracker.sv_full,1)<500
            svm_tracker.sv_full = sample(mask_pos | mask_neg,:);
            svm_tracker.sv_label = label(mask_pos | mask_neg,:);
        end
        if config.verbose
            fprintf('psvm: feat_d: %d; train_num: %d; svs: %d \n',size(sample,2),size(sample,1),size(svm_tracker.sv_full,1)); 
        end
    case 5 % tvm
%         figure(4)
%         bar([size(svm_tracker.pos_sv,1),size(svm_tracker.neg_sv,1)]);
        
%         score = -(sample*svm_tracker.clsf.w'+svm_tracker.clsf.Bias);
%         pos_score = score(label>0.5);
%         neg_score = score(label<0.5);
%         mask = (rand(size(label))>0.5); %& ...
%             %(score > max(neg_score) | score<min(pos_score));
%         sample = sample(mask,:);
%         label = label(mask,:);
        num_newsample = size(sample,1);

        sample = [svm_tracker.pos_sv;svm_tracker.neg_sv; sample];
        label = [ones(size(svm_tracker.pos_sv,1),1);zeros(size(svm_tracker.neg_sv,1),1);label];% positive:1 negative:0
        sample_w = [svm_tracker.pos_w;svm_tracker.neg_w;fuzzy_weight];
       
        pos_mask = label>0.5;
        neg_mask = ~pos_mask;
        s1 = sum(sample_w(pos_mask));
        s2 = sum(sample_w(neg_mask));
        
        sample_w(pos_mask) = sample_w(pos_mask)*s2;
        sample_w(neg_mask) = sample_w(neg_mask)*s1;
        
        C = max(svm_tracker.C*sample_w/sum(sample_w),0.001);
        
        % whitening ********************
%         tmp_msk = svm_tracker.pos_w > 1;
%         sigma = sum(svm_tracker.pos_corr(:,:,tmp_msk),3)-sum(svm_tracker.pos_ms(:,:,tmp_msk),3);
%         wht = speye(size(svm_tracker.pos_sv,2))+svm_tracker.lambda*sigma;
%         svm_tracker.whitening = sqrtm(inv(wht));
        
        if config.verbose
            svm_tracker.clsf = svmtrain_my( sample, label, ...%'kernel_function',@kfun,'kfunargs',{svm_tracker.struct_mat},...
                'boxconstraint',C,'autoscale','false','options',statset('Display','final','MaxIter',5000));
            fprintf('feat_d: %d; train_num: %d; sv_num: %d \n',size(sample,2),size(sample,1),size(svm_tracker.clsf.Alpha,1)); 
        else
            svm_tracker.clsf = svmtrain_my( sample, label, ...%'kernel_function',@kfun,'kfunargs',{svm_tracker.struct_mat},...
                'boxconstraint',C,'autoscale','false','options',statset('MaxIter',5000));
        end
        %**************************
        if ~isempty(svm_tracker.w)
            s_rate = svm_tracker.w_smooth_rate;
            svm_tracker.w = s_rate*svm_tracker.w + (1-s_rate)*svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors*svm_tracker.struct_mat;
            svm_tracker.Bias = s_rate*svm_tracker.Bias + (1-s_rate)*svm_tracker.clsf.Bias;
        else
            svm_tracker.w = svm_tracker.clsf.Alpha'*svm_tracker.clsf.SupportVectors*svm_tracker.struct_mat;
            svm_tracker.Bias = svm_tracker.clsf.Bias;
        end
        svm_tracker.clsf.w = svm_tracker.w;
        % get the idx of new svs
        sv_idx = svm_tracker.clsf.SupportVectorIndices;
        sv_old_sz = size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1);
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
            pos_dis_cro = pdist2(svm_tracker.pos_sv,pos_sv_new);
            svm_tracker.pos_dis = [svm_tracker.pos_dis, pos_dis_cro; pos_dis_cro', pos_dis_new];
            svm_tracker.pos_sv = [svm_tracker.pos_sv;pos_sv_new];
            svm_tracker.pos_w = [svm_tracker.pos_w;ones(num_sv_pos_new,1)];
            
            % update structrual information ********************
%             pos_corr = zeros(size(svm_tracker.pos_sv,2),size(svm_tracker.pos_sv,2),...
%                 num_sv_pos_new);
%             for k = 1:num_sv_pos_new
%                 pos_corr(:,:,k) = pos_sv_new(k,:)'*pos_sv_new(k,:);
%             end
%             svm_tracker.pos_corr = cat(3,svm_tracker.pos_corr,pos_corr);
%             svm_tracker.pos_ms = cat(3,svm_tracker.pos_ms,pos_corr);
        end
        
        % update neg_dis, neg_w and neg_sv
        neg_sv_new = sv_new(sv_new_label<0.5,:);
        if ~isempty(neg_sv_new)
            if size(neg_sv_new,1)>1
                neg_dis_new = squareform(pdist(neg_sv_new));
            else
                neg_dis_new = 0;
            end
            neg_dis_cro = pdist2(svm_tracker.neg_sv,neg_sv_new);
            svm_tracker.neg_dis = [svm_tracker.neg_dis, neg_dis_cro; neg_dis_cro', neg_dis_new];
            svm_tracker.neg_sv = [svm_tracker.neg_sv;neg_sv_new];
            svm_tracker.neg_w = [svm_tracker.neg_w;ones(size(sv_new,1)-num_sv_pos_new,1)];
        end
        
        svm_tracker.pos_dis = svm_tracker.pos_dis + diag(inf*ones(size(svm_tracker.pos_dis,1),1));
        svm_tracker.neg_dis = svm_tracker.neg_dis + diag(inf*ones(size(svm_tracker.neg_dis,1),1));
        
        
        % compute real margin
        pos2plane = -svm_tracker.pos_sv*svm_tracker.w';
        neg2plane = -svm_tracker.neg_sv*svm_tracker.w';
        svm_tracker.margin = (min(pos2plane) - max(neg2plane))/norm(svm_tracker.w);
        
        % shrink svs
        % check if to remove
        if size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1)>svm_tracker.B
            pos_score_sv = -(svm_tracker.pos_sv*svm_tracker.w'+svm_tracker.Bias);
            neg_score_sv = -(svm_tracker.neg_sv*svm_tracker.w'+svm_tracker.Bias);
            m_pos = abs(pos_score_sv) < svm_tracker.m2;
            m_neg = abs(neg_score_sv) < svm_tracker.m2;
            
            if config.verbose
                fprintf('remove svs: pos %d, neg %d \n',sum(~m_pos),sum(~m_neg));
            end
            if sum(m_pos) > 0
                svm_tracker.pos_sv = svm_tracker.pos_sv(m_pos,:);
                svm_tracker.pos_w = svm_tracker.pos_w(m_pos,:);
                svm_tracker.pos_dis = svm_tracker.pos_dis(m_pos,m_pos);
            end
            % update structrual information *******************
%             svm_tracker.pos_corr = svm_tracker.pos_corr(:,:,m_pos);
%             svm_tracker.pos_ms = svm_tracker.pos_ms(:,:,m_pos);
            if sum(m_neg)>0
                svm_tracker.neg_sv = svm_tracker.neg_sv(m_neg,:);
                svm_tracker.neg_w = svm_tracker.neg_w(m_neg,:);
                svm_tracker.neg_dis = svm_tracker.neg_dis(m_neg,m_neg);
            end
        end
        
        % check if to merge
        while size(svm_tracker.pos_sv,1)+size(svm_tracker.neg_sv,1)>svm_tracker.B
            [mm_pos,idx_pos] = min(svm_tracker.pos_dis(:));
            [mm_neg,idx_neg] = min(svm_tracker.neg_dis(:));
            
            if mm_pos > mm_neg || size(svm_tracker.pos_sv,1) <= svm_tracker.B_p% merge negative samples
                if config.verbose
                    fprintf('merge negative samples: %d \n', size(svm_tracker.neg_w,1))
                end
                
                [i,j] = ind2sub(size(svm_tracker.neg_dis),idx_neg);
                w_i= svm_tracker.neg_w(i);
                w_j= svm_tracker.neg_w(j);
                merge_sample = (w_i*svm_tracker.neg_sv(i,:)+w_j*svm_tracker.neg_sv(j,:))/(w_i+w_j);                
                
                svm_tracker.neg_sv([i,j],:) = []; svm_tracker.neg_sv(end+1,:) = merge_sample;
                svm_tracker.neg_w([i,j]) = []; svm_tracker.neg_w(end+1,1) = w_i + w_j;
                
                svm_tracker.neg_dis([i,j],:)=[]; svm_tracker.neg_dis(:,[i,j])=[];
                neg_dis_cro = pdist2(svm_tracker.neg_sv(1:end-1,:),merge_sample);
                svm_tracker.neg_dis = [svm_tracker.neg_dis, neg_dis_cro;neg_dis_cro',inf];                
            else
                if config.verbose
                    fprintf('merge positive samples: %d \n', size(svm_tracker.pos_w,1))
                end
%                 if size(svm_tracker.pos_w,1)<4
%                     keyboard
%                 end
                [i,j] = ind2sub(size(svm_tracker.pos_dis),idx_pos);
                w_i= svm_tracker.pos_w(i);
                w_j= svm_tracker.pos_w(j);
                merge_sample = (w_i*svm_tracker.pos_sv(i,:)+w_j*svm_tracker.pos_sv(j,:))/(w_i+w_j);                
                
                % update structrual information *******
%                 pos_corr = (w_i/(w_i+w_j))*svm_tracker.pos_corr(:,:,i) + ...
%                     (w_j/(w_i+w_j))*svm_tracker.pos_corr(:,:,j);
%                 svm_tracker.pos_corr(:,:,[i,j]) = [];
%                 svm_tracker.pos_corr(:,:,end+1) = pos_corr;
%                 svm_tracker.pos_ms(:,:,[i,j]) = [];
%                 svm_tracker.pos_ms(:,:,end+1) = merge_sample'*merge_sample;
                
                
                svm_tracker.pos_sv([i,j],:) = []; svm_tracker.pos_sv(end+1,:) = merge_sample;
                svm_tracker.pos_w([i,j]) = []; svm_tracker.pos_w(end+1,1) = w_i + w_j;
                
                svm_tracker.pos_dis([i,j],:)=[]; svm_tracker.pos_dis(:,[i,j])=[];
                pos_dis_cro = pdist2(svm_tracker.pos_sv(1:end-1,:),merge_sample);
                svm_tracker.pos_dis = [svm_tracker.pos_dis, pos_dis_cro;pos_dis_cro',inf]; 
                
                
            end
            
        end
        
        % update experts
        svm_tracker.experts{end}.w = svm_tracker.w;
        svm_tracker.experts{end}.Bias = svm_tracker.Bias;
     
end



% function c = kfun(a,b,m)
% 
% c = a*m*b';
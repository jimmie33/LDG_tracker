function result = MEEMTrack(input, ext, init_rect, start_frame, end_frame)

addpath(genpath('.'));
addpath('../../mexopencv/mexopencv');

% parse input arguments
D = dir(fullfile(input,['*.', ext]));
file_list={D.name};

if nargin < 3
    init_rect = -ones(1,4);
end
if nargin < 4
    start_frame = 1;
end
if nargin < 5
    end_frame = numel(file_list);
end

% declare global variables
global sampler;
global svm_tracker;
global experts;
global config
global finish % flag for determination by keystroke

config.display = true;
sampler = createSampler();
svm_tracker = createSvmTracker();
finish = 0; 
experts = {};

% record_vid = false;
% if record_vid
%     vid = avifile('./output/output3.avi');
% end

% figure(1); set(1,'KeyPressFcn', @handleKey); 
% figure(3); set(3,'KeyPressFcn', @handleKey); 


timer = 0;
result.res = nan(end_frame-start_frame+1,4);
result.len = end_frame-start_frame+1;
result.startFrame = start_frame;
result.type = 'rect';


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%    main loop
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
output = zeros(1,4);
% config.svm_thresh = -0.8;
for frame_id = start_frame:end_frame
    
    if finish == 1
        break;
    end
    
    if ~config.display
        clc
        display(input);
        display(['frame: ',num2str(frame_id),'/',num2str(end_frame)]);
    end
    
    %% read a frame
    I_orig=imread(fullfile(input,file_list{frame_id}));
    
    %% intialization
    if frame_id==start_frame
        
        % crop to get the initial window
        if isequal(init_rect,-ones(1,4))
            assert(config.display)
            figure(1)
            imshow(I_orig);
            [InitPatch init_rect]=imcrop(I_orig);
        end
        init_rect = round(init_rect);
        
        config = makeConfig(I_orig,init_rect);
        svm_tracker.output = init_rect*config.image_scale;
        svm_tracker.output(1:2) = svm_tracker.output(1:2) + config.padding;
        svm_tracker.output_exp = svm_tracker.output;
        
        output = svm_tracker.output;
        
        if config.display && ~isequal(init_rect,-ones(1,4))
            figure(1)
            imshow(I_orig);
        end
    end
        
    %% compute ROI and scale image
    [I_orig]= getFrame2Compute(I_orig);
    I_scale = imresize(I_orig,1/svm_tracker.scale);
    
    %% crop frame
    if frame_id == start_frame
        sampler.roi = rsz_rt(svm_tracker.output,size(I_scale),config.search_roi);
    elseif svm_tracker.confidence > config.svm_thresh
        sampler.roi = rsz_rt(svm_tracker.output,size(I_scale),config.search_roi);
    end
    I_crop = I_scale(round(sampler.roi(2):sampler.roi(4)),round(sampler.roi(1):sampler.roi(3)),:);
    
    %% compute feature images
    [BC F] = getFeatureRep(I_crop,config.hist_nbin);
   
    %% tracking part
    
    tic
    
    if frame_id==start_frame
        initSampler(svm_tracker.output,BC,F,config.use_color);
        train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
        label = sampler.costs(train_mask,1)<config.thresh_p;
        fuzzy_weight = ones(size(label));
        initSvmTracker(sampler.patterns_dt(train_mask,:), label, fuzzy_weight);
        
        if config.display
            figure(1);
            imshow(I_orig);
            rectangle('position',svm_tracker.output,'LineWidth',2,'EdgeColor','b')
        end
    else
        % testing
        if config.display
            figure(1)
            imshow(I_orig);       
            roi_reg = sampler.roi; roi_reg(3:4) = sampler.roi(3:4)-sampler.roi(1:2);
            roi_reg = roi_reg*svm_tracker.scale;
            rectangle('position',roi_reg,'LineWidth',1,'EdgeColor','r');
        end
        if mod((frame_id - start_frame + 1),config.expert_update_interval) == 0% svm_tracker.update_count >= config.update_count_thresh
            updateTrackerExperts;
        end
            
%         % evaluate experts
%         evaluateExperts(BC,config.expert_lambda,10);
%             
%         % try different scales and refine tracking position
%         BC = svmTrackerUpDownSampling(BC,F);

        expertsDo(BC,config.expert_lambda,10);
%         BC = svmTrackerUpDownSampling(BC,F);
        
        if svm_tracker.confidence > config.svm_thresh
            output = svm_tracker.output*svm_tracker.scale;
        end
        % visualize sampling positions
%         pos_train = sampler.state_dt;
%         for k = 1:size(pos_train,1)
%             px = pos_train(k,1)+0.5*pos_train(k,3);
%             py = pos_train(k,2)+0.5*pos_train(k,4);
%             rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','r')
%         end 
        
        
        if config.display
            figure(1)
            text(0,5,num2str(svm_tracker.confidence_exp),'BackgroundColor',[1 1 1]);
            text(0,20,num2str(svm_tracker.confidence),'BackgroundColor',[1 1 1]);
            
%             if svm_tracker.confidence_exp > -0.5
            rectangle('position',output,'LineWidth',2,'EdgeColor','g')         
%             else
%                 rectangle('position',svm_tracker.output_exp*svm_tracker.scale,'LineWidth',2,'EdgeColor','k')
%             end
            if svm_tracker.best_expert_idx ~= numel(experts)
                rectangle('position',svm_tracker.output_exp*svm_tracker.scale,'LineWidth',2,'EdgeColor','r')
                rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','y')   
            else
                rectangle('position',svm_tracker.output*svm_tracker.scale,'LineWidth',2,'EdgeColor','b')
            end
        end
        
 
        %% update svm classifier
        svm_tracker.temp_count = svm_tracker.temp_count + 1;
        
%         if svm_tracker.confidence_exp < 0
%             svm_tracker.confidenc_exp = svm_tracker.confidence;
%             svm_tracker.output_exp = svm_tracker.output;
%             svm_tracker.best_expert_idx = numel(svm_tracker.experts);
%         end

        if svm_tracker.confidence > config.svm_thresh %&& ~svm_tracker.failure
%             svm_tracker.template = 0.95*svm_tracker.template + 0.05*svm_tracker.output_feat;
%             resample(BC);        
            train_mask = (sampler.costs<config.thresh_p) | (sampler.costs>=config.thresh_n);
            label = sampler.costs(train_mask) < config.thresh_p;
            
            skip_train = false;
            if svm_tracker.confidence > 1.0 
                score_ = -(sampler.patterns_dt(train_mask,:)*svm_tracker.w'+svm_tracker.Bias);
                if prod(double(score_(label) > 1)) == 1 && prod(double(score_(~label)<1)) == 1
                    skip_train = true;
%                     disp('skipped training!!!!!!!!!!!!!!!!')
                end
            end
            
            if ~skip_train
                costs = sampler.costs(train_mask);
                fuzzy_weight = ones(size(label));
                fuzzy_weight(~label) = 2*costs(~label)-1;
                updateSvmTracker (sampler.patterns_dt(train_mask,:),label,fuzzy_weight);  
                text(250,0,num2str(svm_tracker.margin),'color','r');
            end
            %% visualize traing sample
%             pos_train = sampler.state_dt(train_mask,:);
%             for k = 1:size(label,1)
%                 px = pos_train(k,1)+0.5*pos_train(k,3);
%                 py = pos_train(k,2)+0.5*pos_train(k,4);
%                 if label(k)>0
%                     rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','g')
%                 else
%                     rectangle('position',[px,py,1,1],'LineWidth',1,'EdgeColor','r')
%                 end
%             end 
        else % clear update_count
            svm_tracker.update_count = 0;
        end
% toc
    end
    
%     figure(2)
%     imshow(F(:,:,1)/255)
    
%     if frame_id> 50
%         pause
%     end
%     display('---------------------')
    
    timer = timer + toc;
    res = output;%svm_tracker.output*svm_tracker.scale;
    res(1:2) = res(1:2) - config.padding;
    result.res(frame_id-start_frame+1,:) = res/config.image_scale;
%    toc
   %% visulize results
    
%    fig=figure(1);
%    if record_vid
%        Fr = getframe(fig);
%        vid = addframe(vid,Fr);
%    end
%    
%    
%    figure(3)
% 
%    subplot(2,2,1)
%    output = svm_tracker.output*svm_tracker.scale;
%    imshow(I_orig(round(output(2)+1:output(2)+output(4)-1),...
%        round(output(1)+1:output(1)+output(3)-1),:));
%    subplot(2,2,2) % visualize svm weight vector
%    svm_w = abs(reshape(svm_tracker.w,sampler.template_size(1),sampler.template_size(2),[]));
%    imagesc(sum(svm_w,3));
%    subplot(2,2,3:4)
%    plot(sum(abs(reshape(svm_w,sampler.template_size(1)*sampler.template_size(2),[])),1),'go')
%    
%    figure(2)
%    temp = [];
%    for k = 1:numel(svm_tracker.experts)
%        temp = [temp;svm_tracker.experts{k}.score(max(1,end-24):end)];
%    end
%    plot(temp'),hold on
%    if ~isempty(temp)
%        plot(temp(end,:),'g')
%    end
%    hold off
end

%% output restuls
result.fps = result.len/timer;






% if record_vid
%     vid = close(vid);
% end
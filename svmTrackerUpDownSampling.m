function I_vf = svmTrackerUpDownSampling(I_vf,I)
% for scale changes
global sampler
global svm_tracker
global config


updateSample(I_vf,I,20,0.05,1);


if config.scale_change && svm_tracker.confidence > 0 &&...
        svm_tracker.confidence == svm_tracker.confidence_exp%%%%%%%%%%%%%
    
    confidence_orig = svm_tracker.confidence;
    scale_orig = svm_tracker.scale;
    output_orig = svm_tracker.output;
    roi_orig = sampler.roi;

    %scale up
    if scale_orig < config.scale_upbound
        svm_tracker.scale = scale_orig*config.scale_step;
        sampler.roi = roi_orig/config.scale_step;
        I_vf_up = imresize(I_vf,1/config.scale_step,'nearest');
        updateSample(I_vf_up,I,100,0.2,config.scale_step);
        
        confidence_up = svm_tracker.confidence;
        scale_up = svm_tracker.scale;
        output_up = svm_tracker.output;
        roi_up = sampler.roi;
    else
        confidence_up = -inf;
    end

    %scale down
    if scale_orig > config.scale_lowbound
        svm_tracker.scale = scale_orig/config.scale_step;
        sampler.roi = roi_orig*config.scale_step;
        svm_tracker.output = output_orig;
        I_vf_down = imresize(I_vf,config.scale_step,'nearest');
        updateSample(I_vf_down,I,100,0.2,1/config.scale_step);
        confidence_down = svm_tracker.confidence;
        scale_down = svm_tracker.scale;
        output_down = svm_tracker.output;
        roi_down = sampler.roi;
    else
        confidence_down = -inf;
    end


% if confidence_up >= confidence_orig
%     keyboard
% end

    if confidence_up > confidence_down && ...
            (confidence_up > confidence_orig + 0.5 && confidence_up > 0.0)%%%%%
        svm_tracker.confidence = confidence_up;
        svm_tracker.output = output_up;
        if svm_tracker.state == 1
            svm_tracker.output_old = output_old_up;
        end
        svm_tracker.scale = scale_up;
        sampler.roi = roi_up;
        I_vf = I_vf_up;
    elseif ~( confidence_down > confidence_up && ...
            confidence_down >confidence_orig + 0.5 && confidence_down > 0.0)
        svm_tracker.confidence = confidence_orig;
        svm_tracker.output = output_orig;
        if svm_tracker.state == 1
            svm_tracker.output_old = output_old_orig;
        end
        svm_tracker.scale = scale_orig;
        sampler.roi = roi_orig;
    else
        I_vf = I_vf_down;
    end
    
    svm_tracker.output_exp = svm_tracker.output;
    svm_tracker.confidence_exp = svm_tracker.confidence;
    
end

%scale down
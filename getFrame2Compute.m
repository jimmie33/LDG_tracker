function [I_orig] = getFrame2Compute(I_orig)
% handle color converstion and resizing
global config

if config.padding >0
        I_orig = padarray(I_orig,[config.padding, config.padding],'replicate');
end

if config.image_scale ~= 1
    I_orig = cv.resize(I_orig,config.image_scale);
    %I = imresize(I_orig,[config.image_scale*size(I_orig,1),config.image_scale*size(I_orig,2)]);
%     I_orig = I;
else
%     I = I_orig;
end



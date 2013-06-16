function [I I_orig] = getFrame2Compute(I_orig)
% handle color converstion and resizing
global config

if config.image_scale ~= 1
    I = imresize(I_orig,[config.image_scale*size(I_orig,1),config.image_scale*size(I_orig,2)]);
    I_orig = I;
else
    I = I_orig;
end
if size(I,3) == 1
    I = double(I)/255;
    config.use_color = false;
elseif config.use_color
    I = RGB2Lab(I);
else
    I = double(rgb2gray(I_orig))/255;
end


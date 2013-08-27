function mask = getEllipseMask(mask_sz)

sz = max(mask_sz(1:2));

[x y] = meshgrid(1:sz,1:sz);
xc = ceil((sz+1)/2);
yc = ceil((sz+1)/2);
d = ((x-xc).^2 + (y-yc).^2) < xc^2;
mask = zeros(sz,sz);
mask(sub2ind(size(mask),x(d),y(d))) = 1;
mask = imresize(mask,mask_sz(1:2), 'nearest') > 0;
if numel(mask_sz) > 2
    mask = repmat(mask,[1 1 mask_sz(3)]);
end

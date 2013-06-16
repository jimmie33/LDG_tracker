function rect = rsz_rt(rect,bd_sz,scale)
r = max(rect(3:4));
rect = round([max([rect(1)+0.5*rect(3)-0.5*scale*r,1]),max([rect(2)+0.5*rect(4)-0.5*scale*r,1]),...
           min([rect(1)+0.5*rect(3)+0.5*scale*r,bd_sz(2)]),min([rect(2)+0.5*rect(4)+0.5*scale*r,bd_sz(1)])]);
function iou = getIOU(r1,r2)

left = max(round(r1(:,1)),round(r2(:,1)));
top = max(round(r1(:,2)),round(r2(:,2)));
right = min(round(r1(:,1)+r1(:,3)),round(r2(:,1)+r2(:,3)));
bottom = min(round(r1(:,2)+r1(:,4)),round(r2(:,2)+r2(:,4)));
ovlp = max(right - left,0).*max(bottom - top, 0);
iou = ovlp./(r1(:,3).*r1(:,4)+r2(:,3).*r2(:,4)-ovlp);

end
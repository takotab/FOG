function [xvert,yvert] = ParabolaVertex(x,y)

if numel(x)~=3 || numel(y)~=3 || numel(unique(x))~=3
    error ('x and y must be 3-element vectors, and x must contain 3 unique elements');
end

abc = [x(:).^2 x(:) ones(3,1)]\y(:);

xvert = -abc(2)/abc(1)/2;
if nargout>1
    yvert = [xvert.^2 xvert 1]*abc;
end
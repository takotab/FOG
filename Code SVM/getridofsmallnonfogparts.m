function [fogtime,cFOG] = getridofsmallnonfogparts(fogtime,cFOG,time_too_small)
%getridofsmallnonfogparts checks whether there are fogtime inputs that are
%to small and deletes these
%if start current input + time_too_small is less than the end current input then:
%       delete input
%

if nargin == 1
    cFOG = 0;
end
if nargin == 2
    time_too_small = 1;
end
i = 1;
while i < size(fogtime.time,1)
    
    if fogtime.time(i,1)+time_too_small > fogtime.time(i,1)
        fogtime.time(i,:) = [];
        fogtime.event(i) = [];
        cFOG = cFOG-1;
    end
    i = i+1;
end

end


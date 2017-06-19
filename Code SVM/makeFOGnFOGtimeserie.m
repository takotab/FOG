function [ FOGnFOG ] = makeFOGnFOGtimeserie( fogtime,time )
%makeFOGnFOGtimeserie transforms the struct fogtime to a timeserie
%depicting the current state. these are states and there corresponding
%states:
%
%   fogwalk = 1
%   fogstop = 1.1
%   Cogstop = 0.7
%   nfog = 0
%
% tako tabak feb 2017
FOGnFOG = zeros(length(time),1);
curr_FOG_n = 1;
FOG_now = 0;
i =1;
[r,~] = size(fogtime.time);
while curr_FOG_n <= r
    
    if fogtime.time(curr_FOG_n,1) <= time(i)
        FOG_now = change_FOGnFOG(fogtime.event{curr_FOG_n}); % there is now FOG or something else
        
    end
    if fogtime.time(curr_FOG_n,2) <= time(i)
        
        
        FOG_now = 0; % the FOG stoped
        
        curr_FOG_n = curr_FOG_n+1;
    end
    FOGnFOG(i) = FOG_now;
    i=i+1;
end
end

function [returnvalue] = change_FOGnFOG(event)
switch event
    case 'FreezeWalk'
        
        returnvalue = 1;
        
    case 'FreezeTurn'
        
        returnvalue = 1.1;
        
    case 'CogStop'
        
        returnvalue = 0.7;
        
    case ''
        
        returnvalue = 0;
        
        
end

end


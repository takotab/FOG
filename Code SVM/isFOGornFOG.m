function [ boolian, class_if_more_classes ,fog_window_stats] = isFOGornFOG(currentwindowFOG)
%isFOGornFOG calculates wheter the window should be fog or not and
%calculates some statistics over the this window
% fog_window_stats(1) -> total sum of the window
% fog_window_stats(2) -> the percenatage fogwalk
% fog_window_stats(3) -> the percenatage fogstop
% fog_window_stats(4) -> the percenatage cogstop
% fog_window_stats(5) -> the final fog or nfog
% fog_window_stats(6) -> the class if there were multiple classes 

%   Detailed explanation goes here
if nargin<1
    currentwindowFOG = [1.1 1.1 0 1.1 1.1 1.1 0 0 1.1 1.1 0 0 0.7 0 0 1.1 1.1 1.1 1.1];
end
if nargout == 2
fog_window_stats = [sum(currentwindowFOG)/length(currentwindowFOG),(sum(currentwindowFOG == 1)/length(currentwindowFOG)),(sum(currentwindowFOG == 1.1)/length(currentwindowFOG)),(sum(currentwindowFOG == 0.7)/length(currentwindowFOG)),0,(sum(currentwindowFOG == 0)/length(currentwindowFOG))];
[max_result,fog_window_stats(6)] = max(fog_window_stats(2:4));
if max_result < 0.5
    fog_window_stats(6) = 0;
end
% fog_window_stats(6) :
%   1 = fogwalk
%   2 = fogstop
%   3 = Cogstop 
%   5 = nfog
end
currentwindowFOG(currentwindowFOG == 1.1) = 1;
currentwindowFOG(currentwindowFOG == 0.7) = 0;

halveN = length(currentwindowFOG)/2;
boolian = 0;
if sum(currentwindowFOG) > halveN
    boolian = 1;
    fog_window_stats(5) = boolian;
end

class_if_more_classes = fog_window_stats(6);

end


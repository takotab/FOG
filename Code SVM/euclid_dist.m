function [ acceldist ] = euclid_dist( accel )
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
acceldist = nan(size(accel,1),1);
for i = 1:length(accel)
    acceldist(i) = (accel(i,1)^2 +accel(i,2)^2+accel(i,3)^2)^(1/2);
    
end

end


function [ahlrichsfeatures] = alhrichsfeatures(psd,f)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

locomotionbandindex = find(f>=0.5 & f<3);
freezebandindex = find(f>=3 & f<=8);
ahlrichsfeatures(1,1:2) = [mean(psd(freezebandindex)),mean(psd(locomotionbandindex))];                %mean amplitude
ahlrichsfeatures(1,3:4) = [std(psd(freezebandindex)),std(psd(locomotionbandindex))];                  %std amplidute
ahlrichsfeatures(1,5:6) = [entropy_t(psd(freezebandindex)),entropy_t(psd(locomotionbandindex))];      %entropy amplitude
[ahlrichsfeatures(1,7),i_freeze] = max(psd(freezebandindex));                                         %peak amplitude freeze band
[ahlrichsfeatures(1,8),i_loco] = max(psd(locomotionbandindex));                                       %peak amplitude loco band
ahlrichsfeatures(1,9:10) = [f(freezebandindex(1)+i_freeze-1),f(locomotionbandindex(1)+i_loco-1)];     %frequency of the peak
ahlrichsfeatures(1,11:15) = freezeindexcap(f,psd,freezebandindex,locomotionbandindex);
end

function [returnvalue] = freezeindexcap(f,psd,freezebandindex,locomotionbandindex)

sum_freeze =  x_numericalIntegration(psd(freezebandindex),100);
sum_loco =  x_numericalIntegration(psd(locomotionbandindex),100);
fi = sum_freeze/sum_loco;
bachlin = sum_loco + sum_freeze;
stepcadence = find2thpeak(psd,f);
stepcadence_prev = NaN;
stepcadence_prev_prev = NaN;
returnvalue = [fi,bachlin,stepcadence,stepcadence_prev,stepcadence_prev_prev];
end
function i = x_numericalIntegration(x,SR)
%
% Do numerical integration of x with the sampling rate SR
% -------------------------------------------------------------------
% Copyright 2008 Marc Bächlin, ETH Zurich, Wearable Computing Lab.
%
% -------------------------------------------------------------------
i = (sum(x(2:end))/SR+sum(x(1:end-1))/SR)/2;
end
function stepcandence = find2thpeak(psd,f)
[~,locs] = findpeaks(psd);

if isempty(locs) || size(locs,1) == 1
    stepcandence = 0;
elseif isempty(locs(2))
    stepcandence = f(locs(1))*2;
else
    stepcandence = f(locs(2));
end



% figure
% plot(f,psd)
% hold on
% plot(stepcandence,psd(i_max +neg_i(1)-1),'*m')

end
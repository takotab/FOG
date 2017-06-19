function [psd_bins] = bins(psd,f,num_bins)
% Returns the the intergral of the first num_bins frequencys
% so num_bins = 2 size(psd_bins)= [1 2]
% with the energy form 0 to 1 and 1 to 2
    if nargin==2
        num_bins = 16;
    end
    
    k = 0;
    psd_bins = nan(1,num_bins);
    for j = 1:num_bins
        index = find(f>=k & f<j);
        psd_bins(j) = trapz(f(index),psd(index));
        if psd_bins(j) > 2e+7
            warning('something went wrong ')
            
        end
        k=j;
    end
end
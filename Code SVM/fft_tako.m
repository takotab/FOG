function [ psd,f ] = fft_tako( accel_norm, sample_frequency)
%fft_tako Returns the power spectral density of the signal and the corresponding frequencies

accel_norm = accel_norm - mean(accel_norm);

if nargin == 1 || sample_frequency == 100
    psd = abs(fft(accel_norm).^2)/length(accel_norm);
    sample_frequency =100;
else
    psd = abs(fft(accel_norm).^2)/length(accel_norm);
end
f = [0:length(accel_norm)-1]*(1/(length(accel_norm)*(1/sample_frequency)));

%f = [0:length(accel_norm)]*(1/(length(accel_norm)*(1/sample_frequency)));
end





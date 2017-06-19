function [outcome_MBTM] = eval_MBTM(featurematrix,MBTM)
if nargin ==0 
    load('eval_MBTM_save.mat');
end
if nargin < 2
    MBTM.Freeze_index_threshold = 200;
    MBTM.Energie_threshold = 2;
end

outcome_MBTM = zeros(size(featurematrix,1),1);
%go over every datasamples and determ outcome of the MBTM algorithm

outcome_MBTM( featurematrix(:,44) > MBTM.Freeze_index_threshold & ...
    featurematrix(:,45) > MBTM.Energie_threshold) = 1 ;


end
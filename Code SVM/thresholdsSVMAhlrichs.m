function [PredictedYtest] = thresholdsSVMAhlrichs(PredictedYtest_SVM,input,forwards_or_backwards)
%24012017
if nargin ==0
    clear all
   load('test_outcome.mat') 
   PredictedYtest_SVM = outcome;
end
if isempty(input)
   input = [60 0.45 0.55]; 
end
if nargin < 3
    forwards_or_backwards = 'forwards';
end
%fprintf('Starting variation %i\n',length(th)+1)


[n,~] = size(PredictedYtest_SVM);
if nansum(input(1)) >= n
    PredictedYtest = PredictedYtest_SVM;
else
    if strcmp(forwards_or_backwards,'forwards')
        %% just forwards
        % constants
        m = round(input(1));
        th = input(2:end);        
        
        cj = nan(length(PredictedYtest_SVM)-m,1);
        PredictedYtest = cj;
        
        % because you can not check te confidence of the last m classifications
        % these will just be the same as done by the SVM
        
        PredictedYtest(n-m:n) = round(PredictedYtest_SVM(n-m:n));
        for i = 1:length(PredictedYtest_SVM)-m
            %% go over the samples
            % calculate the confidence
            cj(i) = (1/m)*sum(PredictedYtest_SVM(i:i+m-1));
            
            %classify based on the threshold(s)
            PredictedYtest(i) = classify_usingthresholds(cj(i),th);
        end
        
        
    elseif strcmp(forwards_or_backwards,'backwards')
        %% backwards
        m = round(input(1));
        th = input(2:end); 
        
        cj = nan(length(PredictedYtest_SVM)-m,1);
        
        PredictedYtest = cj;
        
        % because you can not check te confidence of the first m classifications
        % these will just be the same as done by the SVM
        
        PredictedYtest(1:m) = PredictedYtest_SVM(1:m);
        
        for i = m:length(PredictedYtest_SVM)
            %% go over the samples
            % calculate the confidence
            cj(i) = (1/m)*sum(PredictedYtest_SVM(i-m+1:i));
            
            %classify based on the threshold(s)
            PredictedYtest(i) = classify_usingthresholds(cj(i),th);
        end
        
        
        
    else
        %% both
        m = round(input(1:2));
        th = input(3:end);  
        cj = nan(length(PredictedYtest_SVM),1);
        
        PredictedYtest = cj;
        
        % because you can not check te confidence of the first m classifications
        % these will just be the same as done by the SVM
        
        PredictedYtest(1:m(2)) = PredictedYtest_SVM(1:m(2));                
        PredictedYtest(n-m(1):n) = PredictedYtest_SVM(n-m(1):n);
        
        for i = m(2):length(PredictedYtest_SVM)-m(1)
            %% go over the samples
            % calculate the confidence
            cj(i) = (1/sum(m))*sum(PredictedYtest_SVM(i-m(2)+1:i+m(1)-1));
            
            %classify based on the threshold(s)
            PredictedYtest(i) = classify_usingthresholds(cj(i),th);
            
        end
        
    end
    
end
if length(PredictedYtest) ~= length(PredictedYtest_SVM)
    warning('The predicted Y using thresholds does not have the same size as original')
end
end


function PredictedYtest = classify_usingthresholds(cj,th)
PredictedYtest = nan;
if length(th) == 1 % variation 2
    if cj > th
        PredictedYtest = 1;  % Freeze
    else
        PredictedYtest = 0;  % non-Freeze
    end
elseif length(th) == 2 % variation 3
    if cj < th(1)
        PredictedYtest = 0;  % non-Freeze
    elseif cj >= th(2)
        PredictedYtest = 1;  % Freeze
    else
        PredictedYtest = nan;  % Undefined
    end
else
    warning('length(t) > 3')
end
end


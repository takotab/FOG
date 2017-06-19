function [SplitStruct] = splitdata(X, Y, Ratio,sampleweigths, oldversion, FOGtime)
%splitdata Splits the data into 3 or 2 sets keeping the prior of the 2
%classes. And splits the data set according to the ratio defined in Ratio.
%   Input:
%           X:          an array of N by D with the input data. With the
%                       last colunm being an id of the patient
%           Y:          an vector of length N with the corresponding
%                       classes.
%           Ratio:      an vector of length 2 or 3. In the order Train,
%                       Validation and, Test set.
%
%   Output:
%           SplitStruct with the following content:
%
%           TrainX:     an array of N*Ratio(1) by D with randomly selected
%                       input data.
%           TrainY:     an vector of N*Ratio(1) with the corresponding
%                       classes.
%           ValX:       an array of N*Ratio(2) by D with randomly selected
%                       input data.
%           ValY:       an vector of N*Ratio(2) with the corresponding
%                       classes.
%           TestX:      an array of N*Ratio(3) by D with randomly selected
%                       input data. Only present when length(Ratio) == 3.
%           TestY:      an vector of N*Ratio(2) with the corresponding
%                       classes. Only present when length(Ratio) == 3.

% current problem:
%
% So the samples for the training validation and test sets are randomly
% choicen when you would want to examen them in the right order you can
% not do this within one class because one class will pieces these are
% in the other classes.

% current solution:
%
% Do not randomize the order and accept the different priors in
% different datasets.
%

if nargin == 0
    load('splitdata_test.mat')
end

if ~exist('X','var')
    featurematrix = readtable('37213x73_08022017_PHI.txt')  ;
    X = featurematrix{:,2:73};
    Y = featurematrix{:,1};
    Ratio = [0.6 0.2 0.2];
end


if nargin == 2
    Ratio = [0.6 0.2 0.2];
end

X(isnan(Y),:) = [];
Y(isnan(Y)) = [];


%% Constants and checks
rng(1) %set split to the same seed to get the same splits
[n, ~] = size(X);
classes = unique(Y);
% if size(X,2) == 18 || size(X,2) == 34 || size(X,2) == 50 || size(X,2) == 76
    id_pp = unique(X(:,end));
    fprintf('\nX has %i features (that is %i + 1 + pp_idd(%i) = %i)\n',size(X,2)-2,size(X,2)-2,length(id_pp),size(X,2))
% else
%     warning('please provide patient id with data')
%     id_pp = 1;
%     X(:,end) = ones(size(X,1),1);
%     fprintf('\nX has %i features (that is %i + 1 + pp_idd(%i) = %i)\n',size(X,2)-2,size(X,2)-2,length(id_pp),size(X,2))
% end

if sum(Y) == 0 && length(classes) == 1 % only 1 class present
    error('Only one class present in this Y (eg sum(Y) == 0)')
end
nofsets = length(Ratio);
prior = [length(find(Y==classes(1))), length(find(Y==classes(2)))] ./n;


% if sum(prior) ~= 1
%     error('Priors do not sum to 1')
% end
if length(Ratio) < 2 || length(Ratio) > 3
    error('Ratio vector must have 2 or 3 values.')
elseif sum(Ratio) ~= 1 && sum(Ratio) ~= n
    error('The sum of the vector Ratio must be 1 or equal to n')
elseif n ~= length(Y)
    error('X and Y must have the same length')
elseif length(classes) ~=2
    error('Y can only be of 2 class')
end % data is correct



%% correct and make Ratio and RatioN
if exist('sampleweigths','var')   && length(unique(sampleweigths))>1
    [SplitStruct] = splitdata_sampleweight(X, Y, nofsets, prior, sampleweigths,  Ratio);
    SplitStruct = move_pp_id_from_X_to_new_array(SplitStruct);
    return
end
SplitStruct = struct();
SplitStruct.TrainX = [];
SplitStruct.TrainY = [];

SplitStruct.ValX = [];
SplitStruct.ValY = [];

if nofsets == 3
    SplitStruct.TestX = [];
    SplitStruct.TestY = [];
end
%%
%do different if fogtime is avilible because then the timeserie can keep
%the sequece and dividing the fogtime table is much easier also only do it
%if you split in 2
if ~isempty(FOGtime) && length(Ratio) == 2
    SplitStruct = Split_and_put_it_in_a_box(X,Y,SplitStruct,prior,Ratio,nofsets);
    SplitStruct = split_FOGtime(FOGtime,SplitStruct);
else
    % go over every patient and split them in the right set
    for i_pp = 1:length(id_pp)
        idx_pp = X(:,end)==id_pp(i_pp);
        X_pp = X(idx_pp,:);
        Y_pp = Y(idx_pp,:);
        if size(X_pp,1) > 3
            SplitStruct = Split_and_put_it_in_a_box(X_pp,Y_pp,SplitStruct,prior,Ratio,nofsets);
        end
        
    end
end
%%
prior_after(1,:) = [length(find(SplitStruct.TrainY==classes(1))), length(find(SplitStruct.TrainY==classes(2)))] ;
prior_after(2,:) = [length(find(SplitStruct.ValY==classes(1))), length(find(SplitStruct.ValY==classes(2)))] ;
prior_after(3,:) = [length(find(SplitStruct.TestY==classes(1))), length(find(SplitStruct.TestY==classes(2)))];
prior_after = prior_after./sum(prior_after,2);

%% checks if everything went right
if size(SplitStruct.TrainX,1) ~= size(SplitStruct.TrainY,1)
    error('Something in spliting of data did not go well. TrainX and Y are not same length.')
end
if size(SplitStruct.TestX,1) ~= size(SplitStruct.TestY,1)
    error('Something in spliting of data did not go well. TestX and Y are not same length.')
end
if size(SplitStruct.ValX,1) ~= size(SplitStruct.ValY,1)
    error('Something in spliting of data did not go well. ValX and Y are not same length.')
end
if size(SplitStruct.ValX,1)+size(SplitStruct.TestX,1)+size(SplitStruct.TrainX,1) ~= size(X,1)
    warning('Data created')
    
end

SplitStruct = move_pp_id_from_X_to_new_array(SplitStruct);

warning('The current splits do not randomize the order of FOG episode')
%plot(SplitStruct.TrainY)
end
%%
function SplitStruct = Split_and_put_it_in_a_box(X_in,Y,SplitStruct,prior,Ratio,nofsets)
X= X_in;
%% make sure the training set has at least one episode
if sum(Y) > 0.71
    idx_w_FOG = find(Y>0.71);
    
    %put it in the struct
    SplitStruct.TrainX = [SplitStruct.TrainX;X(idx_w_FOG(1),:)];
    SplitStruct.TrainY = [SplitStruct.TrainY;Y(idx_w_FOG(1),:)];
    
    %delete that epoch
    X(idx_w_FOG(1),:) = [];
    Y(idx_w_FOG(1),:) = [];
end
%%
[n, ~] = size(X);

if nofsets == 2
    RatioN(1,:) = [floor(n*Ratio(1)*prior(1)), round(n*Ratio(1)) - floor(n*Ratio(1)*prior(1))];
    RatioN(2,:) = [floor((n-sum(RatioN(1,:)))*prior(1)), n - sum(RatioN(1,:))-floor((n-sum(RatioN(1,:)))*prior(1))];
    %last one sometimes goes wrong but i dont even use it so maybe
    %better set to nan
else
    RatioN(1,:) = [01,floor(n*Ratio(1))];
    RatioN(2,:) = [floor(n*Ratio(1))+1,floor(n*Ratio(1))+1+floor(n*Ratio(2))];
    RatioN(3,:) = [floor(n*Ratio(1))+1+floor(n*Ratio(2))+1,n];
end


if max(max(RatioN))~=size(X)
   warning('max(max(RatioN))~=size(X)') 
end


%% Find indeces and split numbers for priors

if nofsets == 2
    
    SplitStruct.TrainX = [SplitStruct.TrainX;X(1:sum(RatioN(1,:)),:)];
    SplitStruct.TrainY = [SplitStruct.TrainY;Y(1:sum(RatioN(1,:)),:)];
    
    SplitStruct.TestX = [SplitStruct.ValX;X((sum(RatioN(1,:))):end,:)];
    SplitStruct.TestY = [SplitStruct.ValY;Y((sum(RatioN(1,:))):end,:)];
    
    
else
    
    SplitStruct.TrainX = [SplitStruct.TrainX;X(RatioN(1,1):RatioN(1,2),:)];
    SplitStruct.TrainY = [SplitStruct.TrainY;Y(RatioN(1,1):RatioN(1,2),:)];
    
    SplitStruct.ValX = [SplitStruct.ValX;X(RatioN(2,1):RatioN(2,2),:)];
    SplitStruct.ValY = [SplitStruct.ValY;Y(RatioN(2,1):RatioN(2,2),:)];
    
    SplitStruct.TestX = [SplitStruct.TestX;X(RatioN(3,1):end,:)];
    SplitStruct.TestY = [SplitStruct.TestY;Y(RatioN(3,1):end,:)];
    
end



end
%%
function SplitStruct = move_pp_id_from_X_to_new_array(SplitStruct)
%Move the last row of X to a new array in the struct SplitStruct
do_not_check = 1;
%check if the length is right
possible_num_of_rows = [17 28 49 76 73 55];
m = size(SplitStruct.TrainX,2);
if ( sum(m == possible_num_of_rows)==1 ) || do_not_check == 1  %check if m is in possible_num_of_rows
    if do_not_check ==1 && ( sum(m == possible_num_of_rows)~=1 )
        warning('num of rows is not expected. last row is of X in Splitstruct is still deleted')
    end
    %put the id_pp in the new vector in the struct and delete the last row
    %that contains it.
    SplitStruct.Train_idpp = SplitStruct.TrainX(:,end);
    SplitStruct.TrainX(:,end) = [];
    SplitStruct.Val_idpp = SplitStruct.ValX;
    %SplitStruct.ValX(:,end) = [];
    SplitStruct.Test_idpp = SplitStruct.TestX(:,end);
    SplitStruct.TestX(:,end) = [];
    
else
    warning('num of rows is not expected. X in Splitstruct probally still contains id_pp')
end


end
% function SplitStruct = split_FOGtime(FOGtime,SplitStruct)
% split_time = (SplitStruct.TrainX(end,79))+1.28; %because that is the middel so plus 256/2
% idx_trainFOGtime = find(FOGtime(:,1)<split_time,1,'last');
% FOGtime = [FOGtime(1:idx_trainFOGtime-1,:);...
%     FOGtime(idx_trainFOGtime,1),split_time,FOGtime(idx_trainFOGtime,3);...
%     split_time,FOGtime(idx_trainFOGtime,2),FOGtime(idx_trainFOGtime,3);...
%     FOGtime(idx_trainFOGtime+1:end,:)];
%     
% SplitStruct.Train_FOGtime = FOGtime(1:idx_trainFOGtime,:);
% SplitStruct.Test_FOGtime = FOGtime(idx_trainFOGtime+1:end,:);
% end
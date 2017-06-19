function [conf_matrix] = eval_martinmethod(y_, AO, window_size, overlap)
% eval martinmethod return a confmat in the way descripted by
% Rodríguez-Martín et al. 2017 when the input is the true outcome (k by 3)
% and algorithm outcome (n by 3) of (n by 1)
%{
This is the expaition of Martin in his email to me
We perform 2 tables with 3 columns each. A table for the output of the
algorithm (OA) and a table for the label. Every row represents a FoG
episode (label or OA). At the first column, the beginning time of the ith
 FoG episode is annotated, at the second column, ending time is annotated.
Then at the 3rd column of the label table, we annotate if the episode of
the present row temporally matches with any episode of the other table,
is so, we put a 1. This means we have a True Positive. In case we have a 0
in the row of the label table, it means that our algorithm has not been
able to detect it. This means a False Negative. Regarding the second table,
 we perform the same. We annotate a 1 in case our algorithm temporally
matches with the label table. Now we count all the zeros of this table,
they represent a false positive. True negatives are treated differently.
We perform a new vector which is built like a kind of logic negated AND
between the label and the OA. A new vector is determined with different
episodes of TN. However, we resize them, since the patient can be sitting
for more than 15 minutes. That means that we would have only 1 TN episode
in all this whole episode. Thus, we prefer to divide it in 30 seconds part.
 Every 30 seconds is considered a TN. Read please details, episodes shorter
than 5 s are not considered since they are too short. This way we avoid
oversizing the number of TN which is usually reported in other FoG works
that perform the evaluation through windows. This way specificity is much
more reliable.
%}

if nargin == 0
    y_ = [0 4 0
        4 6 1
        5 69 0
        69 72 1
        72 81 0
        81 82 1
        82 92.5 0];
    y_=[y_;y_(:,1)+92.5,y_(:,2)+92.5,y_(:,3)];
%     y_(10,2) = 95.5;
%     y_(11,1) = 95.5;
    AO = [0 0 1 zeros(1,32) 1 1 0 1 zeros(1,5) zeros(1,4) ]';
    AO = [AO;AO];
    window_size = 2.56;
    overlap = 0.25 * 2.56;
end
if nargin == 2
    window_size = 2.56;
    overlap = 0.25 * 2.56;
    warning('window size and overlap are guessed')
end
%% initilaze
if size(AO,2) == 1
    AO = make_3_columns(AO,window_size,overlap) ;
end
if size(unique(y_(:,3)),1) > 2
    y_(:,3) = y_(:,3) > 0.7;
end
y_(:,1:2) = y_(:,1:2) - y_(1,1);
conf_matrix = zeros(2,2);
TN_to_small = 5;
TN_to_big = 30;
AO_copy = AO;
idx_fog_AO = find(AO(:,3)==1);
TN_time_serie = [0:0.001:round(AO(end,2),2);zeros(1,round(AO(end,2)*1000+1))]';
j=1;
%% Data check
if y_(end,2)*0.95 > AO(end,2) || y_(end,2)*1.05 < AO(end,2)
    keyboard
    error('time-series do not have the same length')
end

%% plot
if nargin ==0
    subplot(3,1,1)
    plot(make_FOGnFOGtimeserie( y_ ))
    title('Label')
    subplot(3,1,2)
    plot(make_FOGnFOGtimeserie( AO))
    title('Algorithm output')  
    
end

% TP FN
for i = 1:size(y_,1)
    if y_(i,3) == 1
        TP= 0 ;
        % end of AO is bigger then start of label
        end_label_start_AO = find(AO(idx_fog_AO,2)> y_(i,1));
        % start of AO is smaller then end of label
        start_label_end_AO = find(AO(idx_fog_AO,1) < y_(i,2));
       
        while j <= size(start_label_end_AO,1) 
            if sum(start_label_end_AO(j) == end_label_start_AO) > 0 % TP
                conf_matrix(2,2) = conf_matrix(2,2) +1;
                %to later check false positives
                AO_copy(idx_fog_AO(j),3)= nan;
                TN_time_serie(round(AO_copy(idx_fog_AO(j),1)*1000)+1:round(AO_copy(idx_fog_AO(j),2)*1000),2) =1;
                TN_time_serie(round(y_(i,1)*1000):round(y_(i,2)*1000),2) =1;
                TP =1;
                
            end
            j = j +1;
        end
        % a true positive was not found so it is a FN
        
        if  TP == 0 % FN
            conf_matrix(2,1) = conf_matrix(2,1) +1;
            TN_time_serie(round(y_(i,1)*1000):round(y_(i,2)*1000),2) =1;
        end
        
    end
end
%% count the number of FP
idx_FP = find(AO_copy(:,3) == 1);
conf_matrix(1,2) = size(idx_FP,1);
for i = 1:size(idx_FP,1)
    TN_time_serie(round(AO(idx_FP(i),1)*1000+1):round(AO(idx_FP(i),2)*1000),2) =1;
end
%%  TN
if nargin == 0;subplot(3,1,3);plot(TN_time_serie(1:100:end,2));end

TN_fogtime = makefogtime(TN_time_serie(:,1),TN_time_serie(:,2));
for i = 1 : size(TN_fogtime,1)
    if(TN_fogtime(i,3) == 0)
        time_no_fog = TN_fogtime(i,2) - TN_fogtime(i,1);
        if time_no_fog > TN_to_small %check if it is not to small
            conf_matrix(1,1) = conf_matrix(1,1) + ceil(time_no_fog/TN_to_big);
        end        
    end    
end
end

% end
%%
function outcome = make_3_columns(outcome,window_size,overlap)
if overlap == 0
    outcome(:,3) = outcome(:,1);
    outcome(:,2) = cumsum(ones(size(outcome,1),1)*window_size);
    outcome(:,1) = outcome(:,2) - ones(size(outcome,1),1)*window_size;
else
    outcome_input = outcome;
    outcome = nan(size(outcome_input,1),3);
    for i =1: size(outcome,1)
        outcome(i,:) = [(i-1)*(window_size-overlap), (i-1)*(window_size-overlap)+window_size, outcome_input(i)];
    end
    
end
end

%%
function timeserie = make_FOGnFOGtimeserie(y)
%NOTE must end with zeros
num_of_stamps = floor(10*y(end,2));
timeserie = zeros(num_of_stamps,1); % 10 hz
i_c = 1;
for i = 1:num_of_stamps
    if i > y(i_c,2)*10
        i_c = i_c + 1;
    end
    timeserie(i) = y(i_c,3);
    
end

end
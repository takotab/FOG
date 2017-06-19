function [obj, MBFA_results_training, MBFA_y_predicted] = train_MBFA(obj,SplitStruct)
if nargin == 0
    load('optimize_MBTM_save.mat')
    other.timelimit = 250;
    
    [m,i] = max(SplitStruct.TrainX(:,44));
    SplitStruct.TrainX(i,:) =[];
    SplitStruct.TrainY(i,:) = [];
    SplitStruct.TrainX_nonzscore(:,45) = SplitStruct.TrainX(:,45);
    SplitStruct.TrainSW = abs(ones(size(SplitStruct.TrainY)).*randn(size(SplitStruct.TrainY)));
    other.x_count_eval_method = 0;
    includeSW = 1;
end

options = optimset('MaxFunEvals',200);

functionincludingparameters = @(x_ini)minimizethisfunction_th(x_ini,SplitStruct.TrainX(:,44:45),SplitStruct.TrainY);

x_ini_matrix= [0.0442 -2.6172e-04;
    -0.2022 -0.4283];
feval = 0;
for i = 1:size(x_ini_matrix,1)
    x_ini = x_ini_matrix(i,:);
    [x_star_,feval_] = fminsearch(functionincludingparameters,x_ini,options);
    if feval_ < feval
        x_star = x_star_;
        feval = feval_;
    end
end

if feval > -0.25 
    % try it again only with random initaliztion points and a limited num 
    % of iterations, and something sticks do more iterations
    tic    
    options = optimset('MaxFunEvals',min(100));
    while feval > -0.25 && toc < 20
        x_ini= randn(1,2);
        x_ini(2) = x_ini(2)*0.5;
        [x_star,feval] = fminsearch(functionincludingparameters,x_ini,options);        
    end
    if feval > -0.25
        options = optimset('MaxFunEvals',200);
        [x_star,feval] = fminsearch(functionincludingparameters,x_ini,options); 
        fprintf('\nSearch again for new start point went well: feval= %i\n',feval)
    end
end
obj.Mdl_MBFA.Freeze_index_threshold = x_star(1);
obj.Mdl_MBFA.Energie_threshold = x_star(2);

if nargout == 3
    [MBFA_y_predicted] = eval_MBTM(SplitStruct.TrainX,obj.Mdl_MBFA);
    [MBFA_results_training] = calcresults(MBFA_y_predicted, SplitStruct.TrainY,[],0,0);
end


end

function output = minimizethisfunction_th(x_ini,Xtrain,Ytrain)
MBTM.Freeze_index_threshold = x_ini(1);
MBTM.Energie_threshold = x_ini(2);
[result] = eval_MBFA(Xtrain,MBTM,Ytrain);
output = result.cost;
end
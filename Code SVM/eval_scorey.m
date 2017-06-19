function outcome = eval_scorey(x_ini,scorey)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
if size(scorey ,2) >1
    scorey = scorey(:,2);
end

outcome = zeros(size(scorey,1),1);
outcome(scorey(:,1) > x_ini) = 1;
end


function [cost] = cost_function(result)
cost = -result.geomean;
if result.sensitivity < 0.7
    cost = cost * (0.3 + result.sensitivity);
end
if result.specificity < 0.7
    cost = cost * (0.3 + result.specificity);
end
if isnan(cost) || cost == 0
    cost = nansum([1 ,-result.accuracy]);
end
end
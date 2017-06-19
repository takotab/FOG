function [h] = entropy_t(p)

h = sum(-(p.*(log2(p))));

end
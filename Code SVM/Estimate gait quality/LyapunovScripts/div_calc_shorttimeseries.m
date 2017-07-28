function divergence=div_calc_shorttimeseries(state,ws_sec,fs,period_sec,progress)% calculate local dynamic stability (max lyapunov exponent). % Input: %   state: appropriate state space %   ws: window size over which divergence should be calculated(in seconds)%   fs: sample frequency %   period: ..... (can be dominant period in the signal(in seconds))%   plotje: show a graph.% % Note that in this version ws should be larger than 4*period, as the long% term divergence is calculated from 4*period to ws*fs.% % Output: %   divergence: the divergence curve%   lds: the 2 estimates of the maximum lyapunov exponents (short term and long term)% % Earlier versions by KvS en SMB. Made the routine faster, 14/01/2011, Sietse Rispens% Use less memory to prevent memory error, 01/02/2011, Sietse Rispens% Take mean of the log of divergence instead of log of the mean of divergence, 26/05/2011, Sietse Rispens% Allow non-integer ws_sec and period_sec and change fitting-range, 01/08/2011, Sietse Rispens% Do not try to find neighbours that need to be followed beyond end of time series, 01/11/2012, Sietse Rispens% Exclude neighbours closer than Period in time (instead of period/2), 01/11/2012, Sietse Rispens[m,n]=size(state);ws=round(ws_sec*fs);period=round(period_sec*fs);mcompletewindow = m-ws+1;statecw = state(1:mcompletewindow,:);divergence_sum=zeros(1,ws);divergence_count=zeros(1,ws);diff_state = statecw*0;diff_state_sqr = diff_state;diff_total = zeros(size(diff_state,1),1);for i = 1:mcompletewindow    if ~isnan(state(i,:))        start=round(max([1,i-period+1]));        stop=round(min([mcompletewindow,i+period-1]));        for j=1:n,            diff_state(:,j) = statecw(:,j)-statecw(i,j);            diff_state_sqr(:,j)=diff_state(:,j).^2;        end        diff_total(:,1)=sum(diff_state_sqr,2);        diff_total(start:stop,1)=NaN;        [mini,index]=min(diff_total);        if i+ws>m || index+ws>m            % do not use these data        else            divergence_sum = divergence_sum + log(sqrt(sum((state(i:i+ws-1,:)-state(index:index+ws-1,:)).^2,2)))';            divergence_count = divergence_count + 1;        end    end    if progress > 0        if mod(i,round(progress))==0            if i>round(progress)                fprintf('\b\b\b\b\b\b\b\b\b\b');            end            fprintf('i=%8d', i);        end    endendif progress > 0    fprintf('\n');enddivergence= (divergence_sum./divergence_count);
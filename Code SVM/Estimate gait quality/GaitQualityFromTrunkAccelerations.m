function [MeasuresStruct] = GaitQualityFromTrunkAccelerations(AccData,FS,NBeforeAfter,LegLength,incl_weiss)
if nargin ==0
    load('C:\Users\Tako\OneDrive\Documenten\School\Research Intership\Code\FARAO-toolbox\GaitQuality_tako.mat');
    addpath('C:\Users\Tako\OneDrive\Documenten\School\Research Intership\Code\FARAO-toolbox\Estimate gait quality\LyapunovScripts');
    LegLength = 0.000001;
end
%% Input:
% AccData: Trunk accelerations during locomotion in VT, ML, AP directions
% FS: sample frequency of the AccData
% NBeforeAfter: Number of samples to omit from start and end of time series
% LegLength: Leg length of the subject in meters

%% Output
% MeasuresStruct: structure containing the measures calculated here as
% fields and subfields

%% History
% 2013-05-29:31 (SR): 1) Add rotation-invariant characteristics for:
%                    -standard devition
%                    -index of harmonicity
%                    -low-frequent percentage
%                     2) Update denominator of index of harmonicity for ML to sum(P(1:2:12))
%                     3) Update order of calculations in frequency analysis
%                     3) Keep exitflags from RealignSensorSignal
% 2013-06-05 (SR): Add condition for stride time that autocovariance must
%                  be positive for any possible direction
% 2013-07-10 (SR): Add characteristics tested by Weiss et al. 2013
% 2013-09-04 (SR): Include episodes of exactly WindowLen for Weiss parameters, i.e. >= instead of >
% 2014-09-23 (KS): Add entropy and updated settings

%% Set some parameters
G = 9.81;                        % gravity acceleration, multiplication factor for accelerations
WindowLen = FS*2.5;               % Minimum length for measures estimation and window length for PSD, non-linear measures and legacy stride time measure
StrideFreqGuess = 1.00;          % used to set search range for stride frequency from 0.5*StrideFreqGuess until 2*StrideFreqGuess
StrideTimeRange = [0.4 4.0];     % Range to search for stride time (seconds)
IgnoreMinMaxStrides = 0.10;      % number or percentage of highest&lowest values ignored for imrpoved variability estimation
N_Harm = 20;                     % number of harmonics used for harmonic ratio, index of harmonicity and phase fluctuation
LowFrequentPowerThresholds = ...
    [0.7 1.4];                   % Threshold frequencies for estimation of low-frequent power percentages
Ly_J = 10;                       % Embedding delay (used in Lyapunov estimations)
Ly_m = 7;                        % Embedding dimension (used in Lyapunov estimations)
Ly_FitWinLen = 60;               % Fitting window length (used in Lyapunov estimations Rosenstein's method)
En_m = 5;
En_r = 0.3;

ApplyRealignment = true;

%% Init output variable
MeasuresStruct = struct();

%% Rescale and select walking part of AccData into AccLoco
AccLoco = G*AccData(NBeforeAfter(1)+1:end-NBeforeAfter(2),:);

%% Only do further processing if time series is long enough
% if size(AccLoco,1) < WindowLen
%     warning('Accloco is not long enough')
%     return;
% end

%% Realign sensor data to VT-ML-AP frame
if ApplyRealignment
    [RealignedAcc, ~] = RealignSensorSignalHRAmp(AccLoco, FS);
    AccLoco = RealignedAcc;
end

%% Stride-times measures
% Stride time and regularity from auto correlation
RangeStart = round(FS*StrideTimeRange(1));
RangeEnd = round(FS*StrideTimeRange(2));
[Autocorr3x3,Lags]=xcov(AccLoco,RangeEnd,'unbiased');
AutocorrSum = sum(Autocorr3x3(:,[1 5 9]),2); % This sum is independent of sensor re-orientation, as long as axes are kept orthogonal
Autocorr4 = [Autocorr3x3(:,[1 5 9]),AutocorrSum];
IXRange = (numel(Lags)-(RangeEnd-RangeStart)):numel(Lags);
% check that autocorrelations are positive for any direction,
% i.e. the 3x3 matrix is positive-definite in the extended sense for
% non-symmetric matrices, meaning that M+M' is positive-definite,
% which is true if the determinants of all square upper left corner
% submatrices of M+M' are positive (Sylvester's criterion)
AutocorrPlusTrans = Autocorr3x3+Autocorr3x3(:,[1 4 7 2 5 8 3 6 9]);
IXRangeNew = IXRange( ...
    AutocorrPlusTrans(IXRange,1) > 0 ...
    & prod(AutocorrPlusTrans(IXRange,[1 5]),2) > prod(AutocorrPlusTrans(IXRange,[2 4]),2) ...
    & prod(AutocorrPlusTrans(IXRange,[1 5 9]),2) + prod(AutocorrPlusTrans(IXRange,[2 6 7]),2) + prod(AutocorrPlusTrans(IXRange,[3 4 8]),2) ...
    > prod(AutocorrPlusTrans(IXRange,[1 6 8]),2) + prod(AutocorrPlusTrans(IXRange,[2 4 9]),2) + prod(AutocorrPlusTrans(IXRange,[3 5 7]),2) ...
    );
if isempty(IXRangeNew)
    StrideTimeSamples = Lags(IXRange(AutocorrSum(IXRange)==max(AutocorrSum(IXRange)))); % to be used in other estimations
    MeasuresStruct.StrideTimeSeconds = nan;
    MeasuresStruct.StrideRegularity = nan;
    MeasuresStruct.RelativeStrideVariability = nan;
else
    StrideTimeSamples = Lags(IXRangeNew(AutocorrSum(IXRangeNew)==max(AutocorrSum(IXRangeNew))));
    MeasuresStruct.StrideRegularity = Autocorr4(Lags==StrideTimeSamples,:)./Autocorr4(Lags==0,:); % Moe-Nilssen&Helbostatt,2004
    MeasuresStruct.RelativeStrideVariability = 1-MeasuresStruct.StrideRegularity;
    MeasuresStruct.StrideTimeSeconds = StrideTimeSamples/FS;
end

%% Measures from height variation by double integration of VT accelerations and high-pass filtering
% Zijlstra & Hof 2003, Assessment of spatio-temporal gait parameters from
% trunk accelerations during human walking, Gait&Posture Volume 18, Issue
% 2, October 2003, Pages 1-10
% set parameters
if LegLength ~= 0
    Cutoff = 0.1;
    MinDist = floor(0.7*0.5*StrideTimeSamples);  % Use StrideTimeSamples estimated above
    % Integrate, filter and select vertical component
    [bz,az] = butter(2,20/(FS/2),'low');
    AccLocoLow20 = filtfilt(bz,az,AccLoco);
    Vel = cumsum(detrend(AccLocoLow20,'constant'))/FS;
    [b,a] = butter(2,Cutoff/(FS/2),'high');
    Pos = cumsum(filtfilt(b,a,Vel))/FS;
    PosFilt = filtfilt(b,a,Pos);
    PosFiltVT = PosFilt(:,1);
    if ~ApplyRealignment % Signals were not realigned, so it has to be done here
        MeanAcc = mean(AccLoco);
        VT = MeanAcc'/norm(MeanAcc);
        PosFiltVT = PosFilt*VT;
    end
    % Find minima and maxima in vertical position
    [PosPks,PosLocs] = findpeaks(PosFiltVT(:,1),'minpeakdistance',MinDist);
    [NegPks,NegLocs] = findpeaks(-PosFiltVT(:,1),'minpeakdistance',MinDist);
    NegPks = -NegPks;
    if isempty(PosPks) && isempty(NegPks)
        PksAndLocs = zeros(0,3);
    else
        PksAndLocs = sortrows([PosPks,PosLocs,ones(size(PosPks)) ; NegPks,NegLocs,-ones(size(NegPks))], 2);
    end
    % Correct events for two consecutive maxima or two consecutive minima
    Events = PksAndLocs(:,2);
    NewEvents = PksAndLocs(:,2);
    Signs = PksAndLocs(:,3);
    FalseEventsIX = find(diff(Signs)==0);
    PksAndLocsToAdd = zeros(0,3);
    PksAndLocsToAddNr = 0;
    for i=1:numel(FalseEventsIX),
        FIX = FalseEventsIX(i);
        if FIX <= 2
            % remove the event
            NewEvents(FIX) = nan;
        elseif FIX >= numel(Events)-2
            % remove the next event
            NewEvents(FIX+1) = nan;
        else
            StrideTimesWhenAdding = [Events(FIX+1)-Events(FIX-2),Events(FIX+3)-Events(FIX)];
            StrideTimesWhenRemoving = Events(FIX+3)-Events(FIX-2);
            if max(abs(StrideTimesWhenAdding-StrideTimeSamples)) < abs(StrideTimesWhenRemoving-StrideTimeSamples)
                % add an event
                [M,IX] = min(Signs(FIX)*PosFiltVT((Events(FIX)+1):(Events(FIX+1)-1)));
                PksAndLocsToAddNr = PksAndLocsToAddNr+1;
                PksAndLocsToAdd(PksAndLocsToAddNr,:) = [M,Events(FIX)+IX,-Signs(FIX)];
            else
                % remove an event
                if FIX >= 5 && FIX <= numel(Events)-5
                    ExpectedEvent = (Events(FIX-4)+Events(FIX+5))/2;
                else
                    ExpectedEvent = (Events(FIX-2)+Events(FIX+3))/2;
                end
                if abs(Events(FIX)-ExpectedEvent) > abs(Events(FIX+1)-ExpectedEvent)
                    NewEvents(FIX) = nan;
                else
                    NewEvents(FIX+1) = nan;
                end
            end
        end
    end
    PksAndLocsCorrected = sortrows([PksAndLocs(~isnan(NewEvents),:);PksAndLocsToAdd],2);
    % Find delta height and delta time
    DH = abs(diff(PksAndLocsCorrected(:,1),1,1));
    DT = diff(PksAndLocsCorrected(:,2),1,1);
    % Correct outliers in delta h
    MaxDH = min(median(DH)+3*mad(DH,1),LegLength/2);
    DH(DH>MaxDH) = MaxDH;
    % Estimate total length and speed
    % (Use delta h and delta t to calculate walking speed: use formula from
    % Z&H, but divide by 2 (skip factor 2)since we get the difference twice
    % each step, and multiply by 1.25 which is the factor suggested by Z&H)
    HalfStepLen = 1.25*sqrt(2*LegLength*DH-DH.^2);
    MeasuresStruct.Distance = sum(HalfStepLen);
    MeasuresStruct.WalkingSpeedMean = MeasuresStruct.Distance/(sum(DT)/FS);
    % Estimate variabilities between strides
    StrideLengths = HalfStepLen(1:end-3) + HalfStepLen(2:end-2) + HalfStepLen(3:end-1) + HalfStepLen(4:end);
    StrideTimes = PksAndLocsCorrected(5:end,2)-PksAndLocsCorrected(1:end-4,2);
    StrideSpeeds = StrideLengths./(StrideTimes/FS);
    WSS = nan(1,4);
    STS = nan(1,4);
    for i=1:4,
        STS(i) = std(StrideTimes(i:4:end))/FS;
        WSS(i) = std(StrideSpeeds(i:4:end));
    end
    
    MeasuresStruct.StepLengthMean=mean(StrideLengths);
    
    MeasuresStruct.StrideTimeVariabilityBestEvent = min(STS);
    MeasuresStruct.StrideSpeedVariabilityBestEvent = min(WSS);
    MeasuresStruct.StrideLengthVariability = std(StrideLengths);
    % Estimate Stride time variability and stride speed variability by removing highest and lowest part
    if ~isinteger(IgnoreMinMaxStrides)
        IgnoreMinMaxStrides = ceil(IgnoreMinMaxStrides*size(StrideTimes,1));
    end
    StrideTimesSorted = sort(StrideTimes);
    MeasuresStruct.StrideTimeVariabilityOmitMinMax = std(StrideTimesSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));
    StrideSpeedSorted = sort(StrideSpeeds);
    MeasuresStruct.StrideSpeedVariabilityOmitMinMax = std(StrideSpeedSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));
    StrideLengthsSorted = sort(StrideLengths);
    MeasuresStruct.StrideLengthVariabilityOmitMinMax = std(StrideLengthsSorted(1+IgnoreMinMaxStrides:end-IgnoreMinMaxStrides));
end
%% 'Movement intensity'
MeasuresStruct.StandardDeviation = std(AccLoco,0,1);
MeasuresStruct.StandardDeviation(1,4) = sqrt(sum(MeasuresStruct.StandardDeviation.^2));

%% Measures from power spectral densities
% Get power spectra of detrended accelerations
AccLocDetrend = detrend(AccLoco);
AccVectorLen = sqrt(sum(AccLocDetrend(:,1:3).^2,2));
P=zeros(0,size(AccLocDetrend,2));
for i=1:size(AccLocDetrend,2),
    [P1,~] = pwelch(AccLocDetrend(:,i),hamming(WindowLen),[],WindowLen,FS);
    [P2,F] = pwelch(AccLocDetrend(end:-1:1,i),hamming(WindowLen),[],WindowLen,FS);
    P(1:numel(P1),i) = (P1+P2)/2;
end
dF = F(2)-F(1);

% Calculate stride frequency and peak widths
[StrideFrequency, ~, PeakWidth, MeanNormalizedPeakWidth] ...
    = StrideFrequencyFrom3dAccBosbaan(P, F);
MeasuresStruct.StrideFrequency = StrideFrequency;

% Add sum of power spectra (as a rotation-invariant spectrum)
P = [P,sum(P,2)];
PS = sqrt(P);

% Calculate the measures for the power per separate dimension
for i=1:size(P,2),
    % Relative cumulative power and frequencies that correspond to these cumulative powers
    PCumRel = cumsum(P(:,i))/sum(P(:,i));
    PSCumRel = cumsum(PS(:,i))/sum(PS(:,i));
    FCumRel = F+0.5*dF;
    
    % Derive relative cumulative power for threshold frequencies
    Nfreqs = size(LowFrequentPowerThresholds,2);
    MeasuresStruct.LowFrequentPercentage(i,1:Nfreqs) = interp1(FCumRel,PCumRel,LowFrequentPowerThresholds)*100;
    
    % Calculate relative power of first twenty harmonics, taking the power
    % of each harmonic with a band of + and - 10% of the first
    % harmonic around it
    PHarm = zeros(N_Harm,1);
    PSHarm = zeros(N_Harm,1);
    for Harm = 1:N_Harm,
        FHarmRange = (Harm+[-0.1 0.1])*StrideFrequency;
        PHarm(Harm) = diff(interp1(FCumRel,PCumRel,FHarmRange));
        PSHarm(Harm) = diff(interp1(FCumRel,PSCumRel,FHarmRange));
    end
    
    % Derive index of harmonicity
    if i == 2 % for ML we expect odd instead of even harmonics
        MeasuresStruct.IndexHarmonicity(i) = PHarm(1)/sum(PHarm(1:2:12));
    elseif i == 4
        MeasuresStruct.IndexHarmonicity(i) = sum(PHarm(1:2))/sum(PHarm(1:12));
    else
        MeasuresStruct.IndexHarmonicity(i) = PHarm(2)/sum(PHarm(2:2:12));
    end
    
    % Calculate the phase speed fluctuations
    PhasePerStrideTimeFluctuation = nan(N_Harm,1);
    StrSamples = round(FS/StrideFrequency);
    for h=1:N_Harm,
        CutOffs = [StrideFrequency*(h-(1/3)) , StrideFrequency*(h+(1/3))]/(FS/2);
        if all(CutOffs<1) % for Stride frequencies above FS/20/2, the highest harmonics are not represented in the power spectrum
            [b,a] = butter(2,CutOffs);
            if i==4 % Take the vector length as a rotation-invariant signal
                AccFilt = filtfilt(b,a,AccVectorLen);
            else
                AccFilt = filtfilt(b,a,AccLocDetrend(:,i));
            end
            Phase = unwrap(angle(hilbert(AccFilt)));
            SmoothPhase = sgolayfilt(Phase,1,2*(floor(FS/StrideFrequency/2))+1); % This is in fact identical to a boxcar filter with linear extrapolation at the edges
            PhasePerStrideTimeFluctuation(h) = std(Phase(1+StrSamples:end,1)-Phase(1:end-StrSamples,1));
        end
    end
    MeasuresStruct.FrequencyVariability(i) = nansum(PhasePerStrideTimeFluctuation./(1:N_Harm)'.*PHarm)/nansum(PHarm);
    
    if i<4,
        % Derive harmonic ratio (two variants)
        if i == 2 % for ML we expect odd instead of even harmonics
            MeasuresStruct.HarmonicRatio(i) = sum(PSHarm(1:2:end-1))/sum(PSHarm(2:2:end)); % relative to summed 3d spectrum
            MeasuresStruct.HarmonicRatioP(i) = sum(PHarm(1:2:end-1))/sum(PHarm(2:2:end)); % relative to own spectrum
        else
            MeasuresStruct.HarmonicRatio(i) = sum(PSHarm(2:2:end))/sum(PSHarm(1:2:end-1));
            MeasuresStruct.HarmonicRatioP(i) = sum(PHarm(2:2:end))/sum(PHarm(1:2:end-1));
        end
    end
end


%% Measures tested by Weiss et al. 2013
% Note: They only analyzed 60 second windows and longer, we analyse all
% windows exceeding WindowLen. Effect seems moderate
if incl_weiss
    MinLenWeiss = WindowLen;
    if size(AccLoco,1) >= MinLenWeiss
        % Spectral measures
        N_Windows_Weiss = floor(size(AccLoco,1)/MinLenWeiss);
        N_SkipBegin_Weiss = ceil((size(AccLoco,1)-N_Windows_Weiss*MinLenWeiss)/2);
        ArrayFDClosest = nan(N_Windows_Weiss,4);
        ArrayFDAmpClosest = nan(N_Windows_Weiss,4);
        ArrayFDWidthClosest = nan(N_Windows_Weiss,4);
        ArrayFD = nan(N_Windows_Weiss,4);
        ArrayFDAmp = nan(N_Windows_Weiss,4);
        ArrayFDWidth = nan(N_Windows_Weiss,4);
        ArrayAccRange = nan(N_Windows_Weiss,3);
        for WinNr = 1:N_Windows_Weiss,
            AccWin = AccLoco(N_SkipBegin_Weiss+(WinNr-1)*MinLenWeiss+(1:MinLenWeiss),:);
            clear P;
            PWwin = 200;
            Nfft = 2^(ceil(log(MinLenWeiss)/log(2)+1));
            for i=1:3
                AccWin_i = (AccWin(:,i)-mean(AccWin(:,i)))/std(AccWin(:,i)); % normalize window
                [P(:,i),F]=pwelch(AccWin_i,PWwin,[],Nfft,FS);
            end
            P(:,4) = sum(P,2);
            for i=1:4
                % P_i=P(:,i)/sum(P(:,i))*Nfft/(FS*2*pi); % normalize to relative power per radian
                P_i=P(:,i);
                IXFRange = find(F>=0.5 & F<= 3);
                FDindClosest = IXFRange(find(P_i(IXFRange)==max(P_i(IXFRange)),1,'first'));
                FDClosest = F(FDindClosest);
                FDAmpClosest = P_i(FDindClosest);
                FDindRange = [find(P_i<0.5*FDAmpClosest & F<F(FDindClosest),1,'last'), find(P_i<0.5*FDAmpClosest & F>F(FDindClosest),1,'first')];
                if numel(FDindRange) == 2
                    FDWidthClosest = diff(F(FDindRange));
                else
                    FDWidthClosest = nan;
                end
                FD = FDClosest;
                FDAmp = FDAmpClosest;
                FDWidth = FDWidthClosest;
                if FDindClosest ~= min(IXFRange) && FDindClosest ~= max(IXFRange)
                    VertexIX = [-1 0 1] + FDindClosest;
                    [FD,FDAmp] = ParabolaVertex(F(VertexIX),P_i(VertexIX));
                    FDindRange = [find(P_i<0.5*FDAmp & F<FD,1,'last'), find(P_i<0.5*FDAmp & F>FD,1,'first')];
                    if numel(FDindRange) == 2
                        StartP = P_i(FDindRange(1)+[0 1]);
                        StartF = F(FDindRange(1)+[0 1]);
                        StopP = P_i(FDindRange(2)-[0 1]);
                        StopF = F(FDindRange(2)-[0 1]);
                        FDRange = [interp1(StartP,StartF,0.5*FDAmp) , interp1(StopP,StopF,0.5*FDAmp)];
                        FDWidth = diff(FDRange);
                    end
                end
                ArrayFDClosest(WinNr,i) = FDClosest;
                ArrayFDAmpClosest(WinNr,i) = FDAmpClosest;
                ArrayFDWidthClosest(WinNr,i) = FDWidthClosest;
                ArrayFD(WinNr,i) = FD;
                ArrayFDAmp(WinNr,i) = FDAmp;
                ArrayFDWidth(WinNr,i) = FDWidth;
            end
            % Temporal measure
            ArrayAccRange(WinNr,:) = max(AccWin)-min(AccWin);
        end
        MeasuresStruct.WeissDominantFreq = [nanmedian(ArrayFDClosest,1);nanmedian(ArrayFD,1)];
        MeasuresStruct.WeissAmplitude = [nanmedian(ArrayFDAmpClosest,1);nanmedian(ArrayFDAmp,1)];
        MeasuresStruct.WeissWidth = [nanmedian(ArrayFDWidthClosest,1);nanmedian(ArrayFDWidth,1)];
        MeasuresStruct.WeissRange = nanmedian(ArrayAccRange,1);
    end
end
%% Non-linear measures
% cut into windows of size WindowLen
N_Windows = floor(size(AccLoco,1)/WindowLen);
N_SkipBegin = ceil((size(AccLoco,1)-N_Windows*WindowLen)/2);
LyapunovW = nan(N_Windows,4);
SE= nan(N_Windows,3);
% LyapunovRC = nan(N_Windows,4);
for WinNr = 1:N_Windows,
    AccWin = AccLoco(N_SkipBegin+(WinNr-1)*WindowLen+(1:WindowLen),:);
    for i=1:3
        [LyapunovW(WinNr,i),~] = CalcMaxLyapWolfFixedEvolv(AccWin(:,i),FS,struct('J',Ly_J,'m',Ly_m));
        [SE(WinNr,i)] = SampleEntropy(AccWin(:,i), En_m, En_r);
        [LyapunovRC(WinNr,i),~] = CalcMaxLyapConvGait(AccWin(:,i),FS,struct('J',Ly_J,'m',Ly_m,'FitWinLen',Ly_FitWinLen));
    end
    Ly_m_allaxes = ceil(Ly_m/size(AccWin,2));
    [LyapunovW(WinNr,4),~] = CalcMaxLyapWolfFixedEvolv(AccWin,FS,struct('J',Ly_J,'m',Ly_m_allaxes));
    [LyapunovRC(WinNr,4),~] = CalcMaxLyapConvGait(AccWin,FS,struct('J',Ly_J,'m',Ly_m_allaxes,'FitWinLen',Ly_FitWinLen));
end
MeasuresStruct.LyapunovW = nanmean(LyapunovW,1);
MeasuresStruct.SampleEntropy=nanmean(SE,1);
MeasuresStruct.LyapunovRC = nanmean(LyapunovRC,1);

if isfield(MeasuresStruct,'StrideFrequency')
    MeasuresStruct.LyapunovPerStrideW = MeasuresStruct.LyapunovW/MeasuresStruct.StrideFrequency;
    MeasuresStruct.LyapunovPerStrideRC = MeasuresStruct.LyapunovRC/MeasuresStruct.StrideFrequency;
end

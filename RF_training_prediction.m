clc;  clear;
Path = '**\A Global Coral-Bleaching Database (GCBD_1998–2020)\';
cd(Path);
file = [Path,'Bleaching_dataset.csv'];
data = readtable(file);
data.mean_Percent_Cover_Sum=[];
data = rmmissing(data);
X_normalized = [data.mean_Latitude_Degrees,data.mean_Longitude_Degrees,  data.mean_Depth_m, data.mean_Windspeed,data.mean_ClimSST,...
    data.mean_Temperature_Kelvin, data.mean_Temperature_Minimum,data.mean_Temperature_Maximum, data.mean_Temperature_Kelvin_Standard_Deviation,...
    data.mean_SSTA, data.mean_SSTA_Minimum, data.mean_SSTA_Maximum, data.mean_SSTA_Standard_Deviation, ...
    data.mean_SSTA_Frequency, data.mean_SSTA_Frequency_Standard_Deviation, data.mean_SSTA_FrequencyMax, ...
    data.mean_SSTA_DHW, data.mean_SSTA_DHWMax,...
    data.mean_TSA, ...
    data.mean_TSA_DHW, data.mean_TSA_DHW_Standard_Deviation, data.mean_TSA_DHWMax];
variable_names = ["Latitude";"Longitude"; "Depth"; "Windspeed";"ClimSST"; ...
    "Temp"; "Temp_min"; "Temp_max"; "Temp_SD";...
    "SSTA"; "SSTA_min"; "SSTA_max"; "SSTA_SD";...
    "SSTA_Frequency"; "SSTA_Frequency_SD"; "SSTA_Frequency_max";...
    "SSTA_DHW"; "SSTA_DHW_max";...
    "TSA"; ...
    "TSA_DHW"; "TSA_DHW_SD"; "TSA_DHW_max"];
lat = round(data.mean_Latitude_Degrees);
lon = round(data.mean_Longitude_Degrees);
% Y = data.avg_bleach;
Y = data.mean_Percent_Bleached_Sum;

%% Traverse the optimal solution
% leaf = 5;  ntrees = 100;  fboot = 1;
% % max_features = size(X, 2);
% max_features = 10;
% best_features = {};
% best_cct = 0;
% tic
% FeatherBar=waitbar(0,'Feature Combinations is Solving...');
% for k = 1:max_features
%     % Obtain the index of the combination
%     feature_combinations = nchoosek(1:size(X, 2), k);
% 
%     for i = 1:size(feature_combinations, 1)
%         selected_features = X(:, feature_combinations(i, :));                     
%         [b,cct, RFRMSE] = Fun_TreeBagger(selected_features, Y,leaf,ntrees,fboot); 
%         if cct > best_cct 
%             best_cct = cct;
%             best_features = feature_combinations(i, :);
%         end
%     end
%     str=['Feature Combinations is Solving...',num2str(100*k/max_features),'%'];
%     waitbar(k/max_features,FeatherBar,str);
% end
% close(FeatherBar);
% toc
% disp(['Best features:' best_features]);
% disp(['Best R^2: ' num2str(best_cct^2)]);

leaf = 5;  ntrees = 100;  fboot = 1;
[b_g,cct_g, RFRMSE_g] = Fun_TreeBaggerCycle(X_normalized,Y,leaf,ntrees,fboot,variable_names);
%% RF Model Storage
% RFModelSavePath='*\RF_Training\';
% save(sprintf('%sRF_f22(Global)_10.mat',RFModelSavePath),'leaf','ntrees','fboot',...
%     'b_g','cct_g','RFRMSE_g');

%% Evaluation of Variable Importance
weights_sta=zeros(10,22);
for i=1:10
[~, ~, ~,weights] = Fun_TreeBaggerCycle(X_normalized,Y,leaf,ntrees,fboot,variable_names);
weights_sta(i,:) = weights;
end
[muHat,sigmaHat,muCI,sigmaCI] = normfit (weights_sta,0.05);
[B,iranked] = sort(muHat,'descend');

%% ------------------------------Predict the global coral bleaching----------------------------------
clc;  clear;
Path = 'EnvironmentalDataPath\';
cd(Path);
filename = 'LoadEnvironmentalData';
T = readtable(filename);
X = [T.lat,T.lon,  T.Depth, T.windspeed_Mean,T.ClimSST...
    T.SST_Mean, T.SST_Min,T.SST_Max, T.SST_StandardDeviation...
    T.SSTA_Mean, T.SSTA_Min,T.SSTA_Max,  T.SSTA_StandardDeviation, ...
    T.SSTA_FrequencyMean, T.SSTA_FrequencyStandardDeviation, T.SSTA_FrequencyMax, ...
    T.SSTA_DHWMean,  T.SSTA_DHWMax,...
    T.TSA_Mean, ...
    T.TSA_DHWMean, T.TSA_DHWStandardDeviation, T.TSA_DHWMax];

normalized_X = NaN(size(X));
nan_rows = any(isnan(X), 2); 
X(nan_rows, :) = []; 
normalized_X(~nan_rows,:) = X;

[bleach_C_idx] = Fun_10Times_RF(normalized_X);
[LN,LT] = meshgrid(-180:1:180,-60:1:60);
bleach_C =  reshape(bleach_C_idx,[length(LN(:,1)),length(LN(1,:))]);
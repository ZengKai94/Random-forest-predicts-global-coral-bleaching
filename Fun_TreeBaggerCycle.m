function [b,cct, RFRMSE,weights] = Fun_TreeBaggerCycle(X,Y,leaf,ntrees,fboot,variable_names)
%% Cycle Preparation
RFScheduleBar=waitbar(0,'Random Forest is Solving...');
RFRMSEMatrix=[];
RFrAllMatrix=[];
RFRunNumSet=50000;
for RFCycleRun=1:RFRunNumSet
    cv = cvpartition(height(X), 'Holdout', 0.1);
    idxTrain = training(cv);     idxTest = test(cv);
    Train_Y = Y(idxTrain,:);     Test_Y = Y(idxTest,:);
    Train_X = X(idxTrain,:);     Test_X = X(idxTest,:);

    %% RF Training
    tic
    disp('Training the tree bagger')

    b = TreeBagger(ntrees, Train_X,Train_Y, 'Method','regression', 'oobvarimp','on', 'surrogate', 'on', 'minleaf',leaf,'FBoot',fboot);
    % RFModel=TreeBagger(nTree,TrainVARI,TrainYield,...
    %     'Method','regression','OOBPredictorImportance','on', 'MinLeafSize',nLeaf);
    %% Test
    disp('Estimate Output using tree bagger')
    [RFPredict_Y,RFPredictConfidenceInterval]=predict(b,Test_X);
    [RFTrain_Y,Y_CI]=predict(b,Train_X);
    % PredictBC107=cellfun(@str2num,PredictBC107(1:end));

    %% Accuracy of RF:calculate the training data correlation coefficient
    RFRMSE=sqrt(sum(sum((RFPredict_Y-Test_Y).^2))/size(Test_Y,1));
    RFrMatrix=corrcoef(RFPredict_Y,Test_Y);
    RFr=RFrMatrix(1,2);
    cct=RFrMatrix(2,1);  % R
    RFRMSEMatrix=[RFRMSEMatrix,RFRMSE];
    RFrAllMatrix=[RFrAllMatrix,RFr];
    if RFRMSE<10 || cct>0.5
         disp(['RMSE:' num2str(RFRMSE),'; R^2:',num2str(cct^2)]);
        break;
    end
    disp(RFCycleRun);
    str=['Random Forest is Solving...',num2str(100*RFCycleRun/RFRunNumSet),'%'];
    waitbar(RFCycleRun/RFRunNumSet,RFScheduleBar,str);
end
close(RFScheduleBar);
toc

% Create a scatter Diagram
disp('Create a scatter Diagram')

figure1 = figure;
axes1 = axes('Parent',figure1,'Position',[0.13 0.17 0.82 0.75]);
hold(axes1,'on');
% plot the 1:1 line
% plot(RFPredict_Y,RFPredict_Y,'LineStyle','-','Color',[0.15 0.15 0.15],'LineWidth',3);

hold on
scatter(Train_Y,RFTrain_Y,'MarkerFaceColor',[0.20,0.65,0.20],'MarkerFaceAlpha',0.7,'MarkerEdgeColor','none');
hold on
scatter(Test_Y,RFPredict_Y,'MarkerFaceColor',[0.89,0.27,0.01],'MarkerFaceAlpha',0.8,'MarkerEdgeColor','none');
% hold off
% grid on

set(gca,'FontSize',18)
xlabel('Actual','FontSize',25)
ylabel('Estimated','FontSize',25)
title(['Validation Dataset, R^2=' num2str(cct^2,2)],'FontSize',30)
% legend('1:1','Training','Validation')

mdl = fitlm(Test_Y,RFPredict_Y);  % Linear regression
x_range = linspace(min(Test_Y), max(Test_Y), 100);
y_range = predict(mdl, x_range');
plot(x_range, y_range,'--', 'DisplayName','Fitted line','LineWidth',3,'Color',[0 0 0]);
plot(x_range, x_range, 'DisplayName','1:1 line','LineWidth',3,'Color',[0 0 0]);
% Plot confidence intervals
[~, ci] = predict(mdl, x_range', 'Prediction', 'curve', 'Alpha', 0.05);
plot(x_range,ci,'--','LineWidth',1.5,'Color','b');
fill([x_range, fliplr(x_range)], [ci(:,1)', fliplr(ci(:,2)')], '-.','DisplayName','95% CI', 'FaceAlpha', 0.3,...
    'EdgeAlpha', 0,'MarkerSize',5, 'MarkerEdgeColor','b','LineWidth',1.5);
box(axes1,'on');
legend('Training','Validation','Fitted line','1:1','95% CI','Position',[0.713437502211891 0.178762379557187 0.228124995576218 0.232673260687601])
drawnow
%--------------------------------------------------------------------------
% Calculate the relative importance of the input variables
tic
disp('Sorting importance into descending order')
weights=b.OOBPermutedVarDeltaError;
[B,iranked] = sort(weights,'descend');
toc

%--------------------------------------------------------------------------
disp(['Plotting a horizontal bar graph of sorted labeled weights.'])

%--------------------------------------------------------------------------
figure
barh(weights(iranked),'g');
xlabel('Variable Importance','FontSize',25,'Interpreter','latex');
ylabel('Variable Rank','FontSize',25,'Interpreter','latex');
title(...
    'Relative Importance of Inputs in estimating Redshift',...
    'FontSize',17,'Interpreter','latex'...
    );
hold on
barh(weights(iranked(1:length(weights))),'y');
barh(weights(iranked(1:3)),'r');

%--------------------------------------------------------------------------
grid on
xt = get(gca,'XTick');
xt_spacing=unique(diff(xt));
xt_spacing=xt_spacing(1);
yt = get(gca,'YTick');
ylim([0.25 length(weights)+0.75]);
xl=xlim;
xlim([0 1.5*max(weights)]);

%--------------------------------------------------------------------------
% Add text labels to each bar
for ii=1:length(weights)
    text(...
        max([0 weights(iranked(ii))+0.02*max(weights)]),ii,...
        variable_names(iranked(ii)),'Interpreter','latex','FontSize',11);
end

%--------------------------------------------------------------------------
set(gca,'FontSize',16)
set(gca,'XTick',0:2*xt_spacing:1.1*max(xl));
set(gca,'YTick',yt);
set(gca,'TickDir','in');
set(gca, 'ydir', 'reverse' )
set(gca,'LineWidth',2);
drawnow

%--------------------------------------------------------------------------
% fn='RelativeImportanceInputs';
% fnpng=[fn,'.png'];
% print('-dpng',fnpng);

%--------------------------------------------------------------------------
% Ploting how weights change with variable rank
disp('Ploting out of bag error versus the number of grown trees')

figure
plot(b.oobError,'LineWidth',2);
xlabel('Number of Trees','FontSize',25)
ylabel('Out of Bag Error','FontSize',25)
title('Out of Bag Error','FontSize',30)
set(gca,'FontSize',16)
set(gca,'LineWidth',2);
grid on
drawnow
% fn='EroorAsFunctionOfForestSize';
% fnpng=[fn,'.png'];
% print('-dpng',fnpng);

end

clear all
close all
warning off

name='balance-scale';

tot_data_struct=importdata([name '\' name '_R.dat']);
index_tune=importdata([name '\conxuntos.dat']);
tot_data=tot_data_struct.data;

% Checking whether any index valu is zero or not if zero then increase all index by 1
if length(find(index_tune == 0))>0
    index_tune = index_tune + 1;
end

% Remove NaN and store in cell
for k=1:size(index_tune,1)
    index_sep{k}=index_tune(k,~isnan(index_tune(k,:)));
end

% Removing first i.e. indexing column and seperate data and classes
data=tot_data(:,2:end);
dataX=data(:,1:end-1);
dataY=data(:,end);
dataYY=dataY; % Just replication for further modifying the class label

% Normalization for each feature
mean_X=mean(dataX,1);
dataX=dataX-repmat(mean_X,size(dataX,1),1);
norm_X=sum(dataX.^2,1);
norm_X=sqrt(norm_X);
norm_eval=norm_X; % Just save fornormalizing the evaluation data
norm_X=repmat(norm_X,size(dataX,1),1);
dataX=dataX./norm_X;
% End of Normalization

unique_classes = unique(dataYY);

% Setting of parameters
OptPara.c1=10^5;
OptPara.theta1=10^4;
OptPara.tau1=0.8;
OptPara.c2=OptPara.c1;
OptPara.tau2=OptPara.tau1;
OptPara.theta2=OptPara.theta1;
OptPara.kerfPara.type='rbf'; 
OptPara.kerfPara.pars=2^-5;
OptPara.max_iter = 10;                    

% For data sets where training-testing partition is not available, performance vealuation is based on cross-validation.
fold_index = importdata([name '/conxuntos_kfold.dat']);

% Checking whether any index value is zero or not if zero then increase all index by 1
if length(find(fold_index == 0))>0
    fold_index = fold_index + 1;
end

for k=1:size(fold_index,1)
    index{k,1}=fold_index(k,~isnan(fold_index(k,:)));
end

for f=1:4
    all_distance=[];
    for mc=1:numel(unique_classes)
        dataY=dataYY;      
        dataY(dataYY==unique_classes(mc),:)=1;
        dataY(dataYY~=unique_classes(mc),:)=-1;
    
        trainX=dataX(index{2*f-1},:);
        trainY=dataY(index{2*f-1},:);
        testX=dataX(index{2*f},:);
        testY=dataY(index{2*f},:);
 
        DataTrain=[trainX trainY];
        test=[testX testY];

        [~,~,~,~,~,~,~,distan]=APLRLSTSVM(test,DataTrain,OptPara);
        all_distance=[all_distance,distan];
    end
     
    % Original labels uses for comparing from predicting value
    trainY_orig=dataYY(index{2*f-1},:);
    testY_orig=dataYY(index{2*f},:);

    [~,Predict_Y]=min(all_distance,[],2);

    if min(unique_classes)==0 && max(unique_classes)==numel(unique_classes)-1
        Predict_Y=Predict_Y-1;
    else
        keyboard
    end
                
    test_acc_tmp(f)=length(find(Predict_Y==testY_orig))/numel(testY_orig);
 
    clear Predict_Y DataTrain trainX trainY testX testY;
end

fprintf('Testing accucracy of APL-RLSTSVM is: %f\n',mean(test_acc_tmp))
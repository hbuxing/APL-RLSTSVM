clear all
close all

% Generate the positive and negative samples
u1=[-1 1];u2=[1 -1];
sigma=[0.15 0;0 0.8];
n_pos=100;
n_neg=100;
X_pos=mvnrnd(u1,sigma,n_pos);
X_neg=mvnrnd(u2,sigma,n_neg);

% Add noise into the positive and negative samples
n_noise=0;
u_noise=[0 0]';
sigma_noise=[1 0.01;0.01 1];
if n_noise>0
    X_noise=mvnrnd(u_noise,sigma_noise,n_noise);
    ind_rand=randperm(n_noise);
    X_noise=X_noise(ind_rand,:);
    X_pos_noise=[X_pos;X_noise(1:n_noise/2,:)];
    X_neg_noise=[X_neg;X_noise(n_noise/2+1:end,:)];
    n_pos_noise=n_pos+n_noise/2;
    n_neg_noise=n_neg+n_noise/2;
end
if n_noise==0
    DataTrain=[X_pos ones(n_pos,1);X_neg -ones(n_neg,1)];
else
    DataTrain=[X_pos_noise ones(n_pos_noise,1);X_neg_noise -ones(n_neg_noise,1)];
end
TestX=[X_pos ones(n_pos,1);X_neg -ones(n_neg,1)];
X_all=[X_pos;X_neg];
X_min=min(X_all)-0.5;
X_max=max(X_all)+0.5;
x_grid=X_min(1):(X_max(1)-X_min(1))/99:X_max(1);
y_grid=X_min(2):(X_max(2)-X_min(2))/99:X_max(2);
[X_grid,Y_grid]=meshgrid(x_grid,y_grid);
X_grid_new=[X_grid(:) Y_grid(:)];
X_grid_new_all=[X_grid_new ones(size(X_grid_new,1),1)];

% Parameter setting for APL-RLSTSVM
FunPara_apl.kerfPara.type='lin';
FunPara_apl.c1=1;
FunPara_apl.theta1=10^4;
FunPara_apl.tau1=sqrt(2)/2;
FunPara_apl.c2=FunPara_apl.c1;
FunPara_apl.theta2=FunPara_apl.theta1;
FunPara_apl.tau2=FunPara_apl.tau1;
FunPara_apl.max_iter=10;


% Testing the training samples without noise
[Predict_Y_tmp,err]=APLRLSTSVM(TestX,DataTrain,FunPara_apl);
Predict_Y=Predict_Y_tmp';

% Show the results
plot(X_all(Predict_Y==1,1),X_all(Predict_Y==1,2),'rx','linewidth',2);
hold on
plot(X_all(Predict_Y==-1,1),X_all(Predict_Y==-1,2),'b.','linewidth',2);
ind_err_apl=find(Predict_Y~=[ones(n_pos,1);-ones(n_neg,1)]);
if n_noise>0
    plot(X_noise(:,1),X_noise(:,2),'k*')
end
plot(X_all(ind_err_apl,1),X_all(ind_err_apl,2),'ks','markersize',10)

% Decision boundary
[~,~,~,~,~,~,~,~,Dec_Val]=APLRLSTSVM(X_grid_new_all,DataTrain,FunPara_apl);
contour(X_grid,Y_grid,reshape(Dec_Val,size(X_grid)),[0 0],'g-','linewidth',2);
title('APL-RLSTSVM')

n_ts=size(TestX,1);
fprintf('APL-RLSTSVM: %f\n',(n_ts-err)/n_ts)

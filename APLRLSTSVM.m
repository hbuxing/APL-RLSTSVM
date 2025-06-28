function [Predict_Y,err,prob_score,A,B,x1,x2,distan,Dec_val]=APLRLSTSVM(TestX,DataTrain,FunPara)
%
% Input: TestX------Testing samples with size (n_test,n_dim+1). The last
%                   column corresponds to class labels.
%        DataTrain--Training samples with size (n_train,n_dim+1). The last
%                   column corresponds to class lables.
%        FunPara----Parameter setting.
%   
% Output: Predict_Y---Predict value of the TestX.
%
% Written by: Hong-Jie Xing
% Date: 2022/12/23

tic;

% Split the training set into the subsets of positive and negative samples
[no_input,no_col]=size(DataTrain);
A=DataTrain(DataTrain(:,end)==1,1:end-1);
B=DataTrain(DataTrain(:,end)==-1,1:end-1);

% Parameter setting
c1=FunPara.c1;
c2=FunPara.c2;
theta1=FunPara.theta1;
theta2=FunPara.theta2;
tau1=FunPara.tau1;
tau2=FunPara.tau2;
max_iter=FunPara.max_iter;
kerfPara=FunPara.kerfPara;

m1=size(A,1);
m2=size(B,1);
e1=ones(m1,1);
e2=ones(m2,1);
epsilon=0.0001;

if strcmp(kerfPara.type,'lin')
    % Initializtion
    H=[A e1];
    G=[B e2];
    w1=1/no_input*ones(no_col-1,1);
    b1=1/no_input;
    w2=w1;
    b2=b1;
    u1=[w1;b1];
    u2=[w2;b2];
    
    lambda2=zeros(m1,1);
    Lambda2=zeros(m1);
    % Iterations
    for k=1:max_iter
        % Compute xi_1 and xi_2
        xi1=B*w1+e2*b1+e2;
        xi2=-(A*w2+e1*b2)+e1;
        % Update Lambda_1 and Lambda_2
        ind_xi1_pos=[];
        ind_xi1_neg=[];
        ind_xi2_pos=[];
        ind_xi2_neg=[];
        % Lambda_1
        lambda1=zeros(m2,1);
        Lambda1=zeros(m2);
        ind_xi1_pos=find(xi1>=0);
        ind_xi1_neg=find(xi1<0);
        lambda1(ind_xi1_pos)=(1+tau1*theta1^2)*(xi1(ind_xi1_pos)+2*tau1)./(xi1(ind_xi1_pos)+tau1).^2;
        lambda1(ind_xi1_neg)=(theta1+(1-theta1)^2*tau1)*(-xi1(ind_xi1_neg)+2*tau1)./(-xi1(ind_xi1_neg)+tau1).^2;
        Lambda1=diag(lambda1);
        % Lambda_2
        lambda2=zeros(m1,1);
        Lambda2=zeros(m1);
        ind_xi2_pos=find(xi2>=0);
        ind_xi2_neg=find(xi2<0);
        lambda2(ind_xi2_pos)=(1+tau2*theta2^2)*(xi2(ind_xi2_pos)+2*tau2)./(xi2(ind_xi2_pos)+tau2).^2;
        lambda2(ind_xi2_neg)=(theta2+(1-theta2)^2*tau2)*(-xi2(ind_xi2_neg)+2*tau2)./(-xi2(ind_xi2_neg)+tau2).^2;
        Lambda2=diag(lambda2);
        % Calculate w_1, b_1, w_2, b_2 
        u1=-inv(G'*Lambda1*G+1/c1*(H'*H)+epsilon*eye(no_col))*G'*Lambda1*e2;
        u2=inv(H'*Lambda2*H+1/c2*(G'*G)+epsilon*eye(no_col))*H'*Lambda2*e1;
    end
    w1=u1(1:end-1);
    b1=u1(end);
    w2=u2(1:end-1);
    b2=u2(end);
    y1=TestX(:,1:end-1)*w1+b1;
    y2=TestX(:,1:end-1)*w2+b2;
else
    % Initialization
    D=DataTrain(:,1:end-1);
    P=[kernelfun(A,kerfPara,D) e1];
    Q=[kernelfun(B,kerfPara,D) e2];
    v1=1/no_input*ones(no_input,1);
    b1=1/no_input;
    v2=v1;
    b2=b1;
    % Iterations
    for k=1:max_iter
        % Compute xi_1 and xi_2
        xi1=kernelfun(B,kerfPara,D)*v1+e2*b1+e2;
        xi2=-kernelfun(A,kerfPara,D)*v2-e1*b2+e1;
        % Update Lambda_1 and Lambda_2
        % Lambda_1
        ind_xi1_pos=[];
        ind_xi1_neg=[];
        ind_xi2_pos=[];
        ind_xi2_neg=[];
        lambda1=zeros(m2,1);
        Lambda1=zeros(m2);
        ind_xi1_pos=find(xi1>=0);
        ind_xi1_neg=find(xi1<0);
        lambda1(ind_xi1_pos)=(1+tau1*theta1^2)*(xi1(ind_xi1_pos)+2*tau1)./(xi1(ind_xi1_pos)+tau1).^2;
        lambda1(ind_xi1_neg)=(theta1+(1-theta1)^2*tau1)*(-xi1(ind_xi1_neg)+2*tau1)./(-xi1(ind_xi1_neg)+tau1).^2;
        Lambda1=diag(lambda1);
        % Lambda_2
        lambda2=zeros(m1,1);
        Lambda2=zeros(m1);
        ind_xi2_pos=find(xi2>=0);
        ind_xi2_neg=find(xi2<0);
        lambda2(ind_xi2_pos)=(1+tau2*theta2^2)*(xi2(ind_xi2_pos)+2*tau2)./(xi2(ind_xi2_pos)+tau2).^2;
        lambda2(ind_xi2_neg)=(theta2+(1-theta2)^2*tau2)*(-xi2(ind_xi2_neg)+2*tau2)./(-xi2(ind_xi2_neg)+tau2).^2;
        Lambda2=diag(lambda2);
        % Calculate v_1, b_1, v_2, b_2
        if m1<m2
            Y=1/epsilon*(eye(m1+m2+1)-P'*inv(epsilon*eye(m1)+P*P')*P);
            Y_til=1/epsilon*(eye(m1+m2+1)-P'*inv(epsilon*inv(Lambda2)+P*P')*P);
            v1_aug=-c1*(Y-Y*Q'*inv(1/c1*inv(Lambda1)+Q*Y*Q')*Q*Y)*Q'*Lambda1*e2;
            v2_aug=(Y_til-Y_til*Q'*inv(c2*eye(m2)+Q*Y_til*Q')*Q*Y_til)*P'*Lambda2*e1;
            v1=v1_aug(1:end-1);
            b1=v1_aug(end);
            v2=v2_aug(1:end-1);
            b2=v2_aug(end);
        elseif m1==m2
            v1_aug=-inv(Q'*Lambda1*Q+1/c1*P'*P+epsilon*eye(no_input+1))*Q'*Lambda1*e2;
            v1=v1_aug(1:end-1);
            b1=v1_aug(end);
            v2_aug=inv(P'*Lambda2*P+1/c2*Q'*Q+epsilon*eye(no_input+1))*P'*Lambda2*e1;
            v2=v2_aug(1:end-1);
            b2=v2_aug(end);            
        else
            Z_til=1/epsilon*(eye(m1+m2+1)-Q'*inv(epsilon*inv(Lambda1)+Q*Q')*Q);
            Z=1/epsilon*(eye(m1+m2+1)-Q'*inv(epsilon*eye(m2)+Q*Q')*Q);
            v1_aug=-(Z_til-Z_til*P'*inv(c1*eye(m1)+P*Z_til*P')*P*Z_til)*Q'*Lambda1*e2;
            v2_aug=c2*(Z-Z*P'*inv(1/c2*inv(Lambda2)+P*Z*P')*P*Z)*P'*Lambda2*e1;
            v1=v1_aug(1:end-1);
            b1=v1_aug(end);
            v2=v2_aug(1:end-1);
            b2=v2_aug(end);
        end
        y1=kernelfun(TestX(:,1:end-1),kerfPara,D)*v1+b1;
        y2=kernelfun(TestX(:,1:end-1),kerfPara,D)*v2+b2;
    end
end

Dec_val=abs(y1)-abs(y2);
Elapse_time=toc;

[no_test,m1]=size(TestX);
if strcmp(kerfPara.type,'lin')
    P_1=TestX(:,1:m1-1);
    y1=P_1*w1+b1;
    y2=P_1*w2+b2;    
else
    C=[A;B];
    P_1=kernelfun(TestX(:,1:end-1),kerfPara,D);
    y1=P_1*v1+b1;
    y2=P_1*v2+b2;
end
for i=1:size(y1,1)
    if (min(abs(y1(i)),abs(y2(i)))==abs(y1(i)))
        Predict_Y(i,1)=1;
    else
        Predict_Y(i,1)=-1;
    end
    dec_bdry(i,1)=min(abs(y1(i)),abs(y2(i)));
end
prob_score=1./(1+exp(-dec_bdry));
[no_test,no_col]=size(TestX);
x1=[]; x2=[]; err=0.;
Predict_Y=Predict_Y';
obs=TestX(:,no_col);
for i=1:no_test
    if(sign(Predict_Y(1,i))~=sign(obs(i)))
        err=err+1;
    end
end  
for i=1:no_test
    if Predict_Y(1,i)==1
        x1=[x1; TestX(i,1:no_col-1)];
    else 
        x2=[x2; TestX(i,1:no_col-1)];
    end
end
distan=abs(y1);
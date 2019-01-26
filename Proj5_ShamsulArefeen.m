%%
close all;
clear all;
No_of_iteration = 30000;
Learning_Rate = 0.005;

% XOR Classification
Data = [0 0; 0 1; 1 0; 1 1]; %XOR Input
t = [0; 1; 1; 0]; %XOR Output
n_Input = size(Data, 2); %No. of Input
n_HiddenLayer = 2; %No of Input in Hidden Layer
n_Output = 2;
l_hidden = 1;
Sample_Size = size(Data,1);
w1 = randn(n_Input, n_HiddenLayer)/sqrt(n_Input); %Weight of layer 1
w2 = randn(n_HiddenLayer, n_Output)/sqrt(n_Output);%Weight of layer 1
b1 = zeros(1,n_HiddenLayer)+0.01; %bias for layer 1
b2 = zeros(1,n_Output)+0.01; %bias for layer 2
y = [t ~t];

for i=1:No_of_iteration
% Forward Pass 
    a1 = (Data * w1) + b1;
    z1 = max(0,a1);
    a2 = (z1 * w2) + b2;
    z2 = exp(a2)./sum(exp(a2),2);
% BackProp  
    delta3 = z2 - y;
    delta_grad2 = (z1')*(delta3);
    b1_bar = sum(delta3,1);
    delta2 = (delta3*(w2')) .* (1*(a1>=0));
    delta_grad1 = (Data')*delta2;
    b2_bar = sum(delta2,1);
% Update
    w1 = w1-Learning_Rate*delta_grad1;
    w2 = w2-Learning_Rate*delta_grad2;
    b1 = b1-Learning_Rate*b2_bar;
    b2 = b2-Learning_Rate*b1_bar;
end

%% XOR Plot
figure(1)
[dat,Y] = meshgrid(-2:0.1:2,-2:0.1:2) ;
X_New = [dat(:) Y(:)];
a1 = X_New*w1+b1;
z1 = max(0,a1);
a2 = z1*w2+b2;
z2 = exp(a2)./sum(exp(a2),2);
[m,n]=max(z2,[],2);
y = ~(n-1);
y = reshape(y,size(dat));
surf(dat,Y,double(y),'FaceAlpha',0.75);
hold on;
scatter(Data(:,1),Data(:,2),[],[1 0 0 1]);
hold off;
title('XOR Classification')

%% REGRESSION
for j = [3 20]
    
rng(100);
dat = (2*rand(1,50)-1)';
tt = (sin(2*pi*dat')+0.3*randn(1,50))';

n_hiddenLayer = 1;% No. of hidden layer
n_Hidden = j; %No. of inputs in hidden layer
Input_n = size(dat,2);
Output_n = 1;
% Experiment = size(dat,1);
% Relaxation = 0;
Weight1 = randn(Input_n,n_Hidden)/sqrt(Input_n);
Weight2 = randn(n_Hidden,Output_n)/sqrt(Output_n);
Bias1 = zeros(1,n_Hidden) + 0.01;
Bias2 = zeros(1,Output_n) + 0.01;

%%
for i=1:No_of_iteration
    Z1 = (dat* Weight1) + Bias1;
    A1 = tanh(Z1);
    Z2 = (A1*Weight2) + Bias2;
    A2 = Z2;
    Del_1 = A2 - tt;
    Del_W1 = (A1')*(Del_1);
    Del_B1 = sum(Del_1,1);
    D2 = (Del_1*(Weight2')) .* (1-A1.^2);
    Del_W2 = (dat')*D2;
    Del_B2 = sum(D2,1);
    Weight1 = Weight1-Learning_Rate*Del_W2;
    Weight2 = Weight2-Learning_Rate*Del_W1;
    Bias1 = Bias1-Learning_Rate*Del_B2;
    Bias2 = Bias2-Learning_Rate*Del_B1;
end

% Regression Plot
figure(2)
if(j==3)
    plot(dat,tt,'gs')
else
    plot(dat,tt,'bs')
end
hold on
[~,indx] = sort(dat(:,1));
sort_mat = [dat(indx,:) A2(indx,:)]; 
if(j==3)
    plot(sort_mat(:,1),sort_mat(:,2),'g-');
else
    plot(sort_mat(:,1),sort_mat(:,2),'b-');
grid on;
end
legend('3','3','20','20')
title('Regression')
end

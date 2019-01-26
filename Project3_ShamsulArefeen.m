clear all;

N = 25; %Sample Size
L = 100; %No. of Experiments
Sigma = 0.3; %Standard Deviation of Noise
S = 0.1; %Given value for S in Gaussian Basis Function
rng(1);
X = rand(1,N); %Random generation of training data in uniform distribution
H =sin(2*pi*X); %Expected value of Target
lambda = linspace(exp(-2.5),exp(2),10); %Allowable value for lambda with linear space
rng(1);
N_test = 1000;
X_test = rand(1,N_test);
H_test =sin(2*pi*X_test);

for m = 1:N-1
    Phi_basis(:,m) = exp(-((X - X(m)).^2)/(2*S^2));
    Phi_basis_test(:,m) = exp(-((X_test - X(m)).^2)/(2*S^2));
end
Phi = [ones(N,1) Phi_basis];
Phi_test = [ones(N_test,1) Phi_basis_test];

I = eye(N);
T = zeros(L,N);
F = zeros(L,N);

for i = 1:length(lambda)
    for j = 1:L
        T(j,:) = H + Sigma*randn(1,N);
        T_test(j,:) = H_test + Sigma*randn(1,N_test);
        t = T(j,:); %Regression
        W(1:N,j) = (inv((Phi'*Phi)+lambda(i)*I))*Phi'*t';
        F(j,:) =Phi* W(:,j); 
        F_test(j,:) =Phi_test* W(:,j);
    end
    
    F_bar = mean(F);
    F_test_bar = mean(F_test);
    T_test_bar = mean(T_test);
    bias_sq(i) = mean((F_bar - H).^2);
    variance(i) =  mean(mean((F - F_bar).^2));
    optimization(i) = bias_sq(i) + variance(i);
    Test_Error(i) = optimization(i)+ mean((F_test_bar - T_test_bar).^2);
end

figure
plot(log(lambda),bias_sq)
hold on
plot(log(lambda),variance)
hold on
plot(log(lambda),optimization)
hold on
plot(log(lambda),Test_Error)
hold off
legend('(bias^2)','variance','(bias^2) + variance','Test Error')
xlabel('ln(\lambda)')



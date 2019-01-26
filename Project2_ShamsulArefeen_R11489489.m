k=1;
for AA = [10 100]
N_train = AA;
rng(19);
X_train = rand(N_train,1);
X_train = sort(X_train);
X_train_norm = X_train./max(X_train);
rng(19);
eps_train = normrnd(0,0.3,N_train,1);
t_train = sin(2*pi*X_train)+eps_train;
t_train_norm=(t_train-min(t_train))/(max(t_train)-min(t_train));
% Test Set Generation
N_test = 100; rng(19);
X_test = rand (N_test,1);
X_test = sort(X_test);
rng(19);
eps_test = normrnd(0,0.3,N_test,1);
t_test = sin(2*pi*X_test)+eps_test;
t_test_norm=(t_test-min(t_test))/(max(t_test)-min(t_test));

%Design Matrix with polynomial for Training Set

X_train_poly = zeros(N_train,9);
for i = 1:N_train
    for m = 1:9
    X_train_poly(i,m) = X_train(i)^m;
    end
end
X_train_dm = [ones(N_train,1) X_train_poly];

%Design Matrix for Test Set

X_test_poly = zeros(N_test,9);
for j = 1:N_test
    for m = 1:9
    X_test_poly(j,m) = X_test(j)^m;
    end
end
X_test_dm = [ones(N_test,1) X_test_poly];

for polyorder=1:10
    theta{polyorder} = pinv(X_train_dm(:,1:polyorder))*t_train_norm;
        Y_train= X_train_dm(:,1:polyorder)*theta{polyorder};
        JW_train = sum((t_train_norm-Y_train).^2);
        MSE_train(polyorder,:) = sqrt(mean((t_train_norm-Y_train).^2));
        E_RMS_train(polyorder,:) = sqrt(JW_train/N_train);
    
        Y_test= X_test_dm(:,1:polyorder)*theta{polyorder};
        Y_test_norm=(Y_test-min(t_test))/(max(t_test)-min(t_test));
        JW_test = sum((t_test_norm-Y_test_norm).^2);
        E_RMS_test(polyorder,:) = sqrt(JW_test/N_test);
        MSE_test(polyorder,:) = sqrt(mean((t_test_norm-Y_test_norm).^2));      
end

%plot
figure(k);
k = k+1;
plot (0:9, E_RMS_train, '-o');
%plot (0:9, MSE_train, '-o');
hold on; grid on;
%plot (0:9, MSE_test, '-x');
plot (0:9, E_RMS_test, '-x');
ylim([0 1]);
xlim([-1 10]);
xlabel('M');
ylabel('E_rms');
legend('Training','Test')
end


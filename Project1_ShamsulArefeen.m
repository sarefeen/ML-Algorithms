load carbig;
Weight(isnan(Horsepower(:,1)),:)=[];
Horsepower(isnan(Horsepower(:,1)),:)=[];
t = Horsepower;
X = [Weight ones(length(Weight),1)];
w = (inv(X'*X))*X'*t;
Closed_Form = X*w;
figure(1);
scatter(Weight,t,'x','r');
xlabel('Weight');
ylabel('Horsepower');
title('Matlab''s Carbig Dataset');
hold on;
z = plot(Weight,Closed_Form,'-b')
legend(z , 'Closed Form');
% Gradient Descent Method
t_norm = t./(max(t));%normalization of data
x_norm = [X(:,1)./(max(Weight)),X(:,2)];
w_guess = [60 20]';
learning_rate = 0.0015;
loop_control = [1;1];
while (loop_control(1) >= 1e-10 && loop_control(2) >= 1e-10)
    old_val= w_guess;
    w_guess = w_guess-learning_rate*(2*(x_norm')*(x_norm*w_guess-t_norm));
    loop_control = abs(w_guess-old_val);
end
y= (x_norm*w_guess).*max(t);
figure(2);
scatter(Weight,t,'x','r');
xlabel('Weight');
ylabel('Horsepower');
title('Matlab''s Carbig Dataset')
hold on;
z1 = plot(Weight,y,'-g');
legend(z1,'Gradient Descent');

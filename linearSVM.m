function testing_output = minSVM(x,y,testing,C,threshold_rate,eps)

%check rate of z convergence
objective_min_value_rate = 9999999999; %update this over time
old_objective_min_value=999999999; %update over time
theta_tilda=ones(1,5); %classifier
z=99999*ones(size(x,1),1); %majorization variable z vector
%create diagonal matrix with last element equal to 0
diagonal=eye(4+1,4+1); %allows us to multiply theta_tilda
diagonal(4+1,4+1)=0;

%continue to go through this while loop until we reach a minimum 
while (objective_min_value_rate > threshold_rate)
    
    %Two parts to finding theta
    %Calculate the sum over the input, x, vectors
    sumX=zeros(4+1,4+1);
    for u=1:size(x,1)
        sumX = sumX + [transpose(x(u,:))*x(u,:)/2/z(u)];
    end
    
    %Calculate the sum over the labels,y, and input vectors, x
    sumY=zeros(4+1,1);
    for u=1:size(x,1)
        sumY = sumY + [(1+z(u)) / 2 / z(u) * y(u) * transpose(x(u,:))];
    end
    
    %Calculate theta_tilda
    theta_tilda=inv(diagonal + C*sumX)*C*sumY;
    
    %Determine the new z basd on theta_tilda
    for t=1:length(z)
        z(t)=max(eps, abs(1-[y(t)*(transpose(theta_tilda)*transpose(x(t,:)))]));
    end
    
    %Find the new value of the majorized objective function, trying to minimize!
    sumMin=0;
    for r=1:size(x,1);
        sumMin=sumMin + [( 1- (y(r)*transpose(theta_tilda)*transpose(x(r,:))) +z(r) )^2 /4/z(r)];
    end
    
    %Check to see if we are at minimum
    new_objective_min_value=0.5*transpose(theta_tilda)*diagonal*theta_tilda+C*sumMin;
    objective_min_value_rate=old_objective_min_value-new_objective_min_value / 1;
    old_objective_min_value=new_objective_min_value;
    intermediate_classification_testing=testing*theta_tilda; %LOOK if you want to see how classification is working

end

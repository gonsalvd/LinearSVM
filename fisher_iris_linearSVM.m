%http://en.wikipedia.org/wiki/Iris_flower_data_set)
%Can get from matlab with 'load fisheriris.mat'

%{
SUMMARY: Implementing a linear SVM implementation and looking at the
effects of a 10%,30% and 50% training with test set (splitting up all the
test data, testing on some to find estimator and then finding
misclassification). 

The linear SVM code here was written mostly off of Dr. Anand's discussion
in class. Here we do a majorization of a new objective function with this
new variable z. This new objective function IS differentiable everything
while the original is NOT due to the hinge loss. We can approximate and get
close to a solution by thus minimizing this new objective function (with z)
by looking at the rate of decrease of the value of the new objective
function and varying parameters like C.

The data set is 150 petal samples from 3 different
kinds of flowers =50 samples/flower. We will be classifying each flower
against each of the other 2 flowers, resulting in 3 comparison pairs
(sat-vergi, sat-versi, vergi-versi). 

We will vary 'C' which weighs vectors that are NOT support vectors. As C
decreases this essentially increases the margin between the two classes
which CAN result in higher misclassification rates BUT also gives you a
more 'generalizable' solution. Too high of a C will mean you did great on
your test data but may do poorly on the real data later on the be
classified.

Varying epsilon, which gave a floor to all the 'z' values seemed to not do
very much to the misclassification.

Varying the size of the training set (increasing size) increased the amount of time to run
the program.

In order to find the minimum of the majorization objective function I
looked for a rate of decrease which got close to 0.
%}

%DESCR: SEPAL LENGHT/SEPAL WIDTH/PETAL LENGTH/PETAL WIDTH
load fisheriris;

C=linspace(0,5,20);
THRESHOLD_RATE_DECREASE=.001; %for us to find the minimum of our new obj function
NUM_TRIALS = 10; %to get an average value
TRAIN_PERCENT = .5; %vary this from 0.1,0.3, to 0.5 (cross validation)
EPS=10^-6; %provides a threshold from zn values due to underflow issues

sat_virg_avg=zeros(NUM_TRIALS,3); %holds average values for plotting
sat_versi_avg=zeros(NUM_TRIALS,3);
virg_versi_avg=zeros(NUM_TRIALS,3);
all_avg=zeros(length(C),10); %allows us to plot C versus each pair set

%Based on the way meas/species is set up
satosa=meas(1:50,:);
versicolor=meas(51:100,:);
virgincia=meas(101:150,:);
total_samples=150;
num_samples_per=50;

err_sat_avg=0;
err_versi_avg=0;
err_virg_avg=0;
err_total_avg=0;

disp('--------START------');

%vary the value of C
for c=1:length(C);
    
    C_VALUE = C(c);
    disp(sprintf('Trials %d Cross-Val %f C %d Eps %f',NUM_TRIALS,TRAIN_PERCENT,C_VALUE,EPS));
    
    for t=1:NUM_TRIALS
        
        %Sample based on our cross validation percentage from satosa,virg,
        %and versi. Randomly choose the training set and keep ratios equal.
        
        sat_train=zeros(TRAIN_PERCENT*num_samples_per,4); %5 x 4
        sat_testing=zeros(50-length(sat_train),4); %45 x 5
        ra = randperm(num_samples_per, TRAIN_PERCENT*num_samples_per); %used to find the TRAINING data
        non_ra=setdiff(linspace(1,50,50),ra); %used to find the TESTING data
        for i=1:length(ra)
            sat_train(i,:)=satosa(ra(i),:);
        end
        for j=1:length(non_ra)
            sat_testing(j,:)=satosa(non_ra(j),:);
        end
        
        versi_train=zeros(TRAIN_PERCENT*num_samples_per,4); %5 x 4
        versi_testing=zeros(50-length(versi_train),4); %45 x 5
        ra = randperm(num_samples_per, TRAIN_PERCENT*num_samples_per);
        non_ra=setdiff(linspace(1,50,50),ra); %used to find the TESTING data
        for i=1:length(ra)
            versi_train(i,:)=versicolor(ra(i),:);
        end
        for j=1:length(non_ra)
            versi_testing(j,:)=versicolor(non_ra(j),:);
        end
        
        virg_train=zeros(TRAIN_PERCENT*num_samples_per,4); %5 x 4
        virg_testing=zeros(50-length(virg_train),4); %45 x 5
        ra = randperm(num_samples_per, TRAIN_PERCENT*num_samples_per);
        non_ra=setdiff(linspace(1,50,50),ra); %used to find the TESTING data
        for i=1:length(ra)
            virg_train(i,:)=virgincia(ra(i),:);
        end
        for j=1:length(non_ra)
            virg_testing(j,:)=virgincia(non_ra(j),:);
        end
        
        
        %pad with 1s to turn theta,theta0 into theta~ (easier)
        sat_train=[sat_train ones(size(sat_train,1),1)];
        virg_train=[virg_train ones(size(virg_train,1),1)];
        versi_train=[versi_train ones(size(versi_train,1),1)];
        
        %DO SVM (majorization linear)
        %Setosa vs. Virgincia
        sat_virg=[sat_train;virg_train]; %make long training vector
        sat_virg_labels=[ones(size(sat_train,1),1); -1.*ones(size(virg_train,1),1)]; %label classes (1,-1) AND pad with 1s
        sat_virg_testing=[sat_testing; virg_testing]; %make long testing set vector
        sat_virg_testing=[sat_virg_testing ones(size(sat_virg_testing,1),1)]; %pad with 1s
        sat_virg_testing_labels=[ones(size(sat_testing,1),1); -1.*ones(size(virg_testing,1),1)]; %label classes
        
        %Setosa vs. Versicolor
        sat_versi=[sat_train;versi_train];
        sat_versi_labels=[ones(size(sat_train,1),1); -1.*ones(size(versi_train,1),1)];
        sat_versi_testing=[sat_testing; versi_testing];
        sat_versi_testing=[sat_versi_testing ones(size(sat_versi_testing,1),1)];
        sat_versi_testing_labels=[ones(size(sat_testing,1),1); -1.*ones(size(versi_testing,1),1)];
        
        %Virgincia vs. Versicolor
        virg_versi=[virg_train;versi_train];
        virg_versi_labels=[ones(size(virg_train,1),1); -1.*ones(size(versi_train,1),1)];
        virg_versi_testing=[virg_testing; versi_testing];
        virg_versi_testing=[virg_versi_testing ones(size(virg_versi_testing,1),1)];
        virg_versi_testing_labels=[ones(size(virg_testing,1),1); -1.*ones(size(versi_testing,1),1)];
        
        %We are doing the linearSVM between 3 pairs
        %(sat-virg,sat-versi,virg-versi)
        sat_virg_testing_output = linearSVM(sat_virg,sat_virg_labels,sat_virg_testing,C_VALUE,THRESHOLD_RATE_DECREASE,EPS);
        sat_versi_testing_output = linearSVM(sat_versi,sat_versi_labels,sat_versi_testing,C_VALUE,THRESHOLD_RATE_DECREASE,EPS);
        virg_versi_testing_output = linearSVM(virg_versi,virg_versi_labels,virg_versi_testing,C_VALUE,THRESHOLD_RATE_DECREASE,EPS);
        
        %add to averages 
        sat_virg_avg(t,:) = findMisclass(sat_virg_testing_output,sat_virg_testing_labels,'sat','virg');
        sat_versi_avg(t,:) = findMisclass(sat_versi_testing_output,sat_versi_testing_labels,'sat','versi');
        virg_versi_avg(t,:) = findMisclass(virg_versi_testing_output,virg_versi_testing_labels,'virg','versi');
        
    end
    % %Output
    disp(sprintf('Avg Misclass Error Prob [sat virg total]: %.2f %.2f %.2f',...
        mean(sat_virg_avg(:,1)),...
        mean(sat_virg_avg(:,2)),...
        mean(sat_virg_avg(:,3))));
    disp(sprintf('Avg Misclass Error Prob [sat versi total]: %.2f %.2f %.2f',...
        mean(sat_versi_avg(:,1)),...
        mean(sat_versi_avg(:,2)),...
        mean(sat_versi_avg(:,3))));
    disp(sprintf('Avg Misclass Error Prob [virg versi total]: %.2f %.2f %.2f',...
        mean(virg_versi_avg(:,1)),...
        mean(virg_versi_avg(:,2)),...
        mean(virg_versi_avg(:,3))));
    
    all_avg(c,:)=[C_VALUE,mean(sat_virg_avg(:,1)),...
        mean(sat_virg_avg(:,2)),...
        mean(sat_virg_avg(:,3)),...
        mean(sat_versi_avg(:,1)),...
        mean(sat_versi_avg(:,2)),...
        mean(sat_versi_avg(:,3)),...
        mean(virg_versi_avg(:,1)),...
        mean(virg_versi_avg(:,2)),...
        mean(virg_versi_avg(:,3))];
end

fig_title=sprintf('NumTrials %d Cross-Val %f Eps %f Thresh-Rate %f',NUM_TRIALS,TRAIN_PERCENT,EPS,THRESHOLD_RATE_DECREASE)
figure('Name',fig_title);
subplot(3,1,1);
plot(all_avg(:,1),all_avg(:,2),...
    all_avg(:,1),all_avg(:,3),...
    all_avg(:,1),all_avg(:,4))
title('sat-virg');
xlabel('C');
ylim([0 .3]);
ylabel('Misclass prob');
legend('Mis1','Mis2','MisTot');

subplot(3,1,2);
plot(all_avg(:,1),all_avg(:,5),...
    all_avg(:,1),all_avg(:,6),...
    all_avg(:,1),all_avg(:,7))
title('sat-versi');
xlabel('C');
ylim([0 .3]);
ylabel('Misclass prob');
legend('Mis1','Mis2','MisTot');

subplot(3,1,3);
plot(all_avg(:,1),all_avg(:,8),...
    all_avg(:,1),all_avg(:,9),...
    all_avg(:,1),all_avg(:,10))
title('virg-versi');
xlabel('C');
ylim([0 .3]);
ylabel('Misclass prob');
legend('Mis1','Mis2','MisTot');

disp('--------END------');

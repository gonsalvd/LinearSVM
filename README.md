Simulink Project: LinearSVM

http://en.wikipedia.org/wiki/Iris_flower_data_set)

SUMMARY: Implementing a linear SVM implementation and looking at the
effects of a 10%,30% and 50% training with test set (splitting up all the
test data, testing on some to find estimator and then finding
misclassification). 

The linear SVM code here was written mostly off of Dr. Anand's discussion
in class. Here we do a majorization of a new objective function with this
new variable z. This new objective function IS differentiable everywhere,
while the original is NOT due to the hinge loss. We can approximate and get
close to a solution by thus minimizing this new objective function (with z).
We can do this by looking at the rate of decrease of the majorization objective value 
and varying parameters like C.

The data set is 150 petal samples from 3 different
types of flowers =50 samples/flower. We will be classifying each flower
against each of the other 2 flowers, resulting in 3 comparison pairs
(sat-vergi, sat-versi, vergi-versi). 

We will vary 'C', which weighs vectors that are not on the correct side of the margin
that are NOT support vectors. As C
decreases this essentially increases the margin between the two classes
which CAN result in higher misclassification rates BUT also gives you a
more 'generalizable' solution. Too high of a C will mean you did great on
your test data but may do poorly on the real data later on the be
classified.

Varying epsilon, which gives a floor to all the 'z' values seemed to not do
very much to the misclassification.

Varying the size of the training set (increasing size) increased the amount of time to run
the program.

In order to find the minimum of the majorization objective function I
looked for a rate of decrease which got close to 0 at about 0.001.
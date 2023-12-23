Introduction


The purpose of this data mining project was to perform classification using Support Vector Machine (SVM), Naïve Bayes, and K-Nearest Neighbors (KNN) algorithms, replacing some values in the dataset with NaN and displaying the result of the replacement, Replace the NaN values you obtained in step 2 with mean of the corresponding column. Display the result of replacement, And Classification on the dataset you obtained in Step 3 with SVM, Naïve Bayes and KNN. Display confusion matrix. The project involved loading a dataset, preprocessing the data, splitting it into training and testing sets, applying feature scaling, training the classifiers, and making predictions on new data points.


Step 1

Data Preprocessing

The dataset used for this project was stored in a CSV file named "Social_Network_Ads.csv". The dataset provided for the classification model contains information on different customers. Each row corresponds to a customer, with features such as age and estimated salary collected by the general manager. The dependent variable is the "purchased" variable, indicating whether the customer has previously bought an SUV from the car company (coded as 1 for purchased, 0 for not purchased). The goal is to train a model to predict which customers are likely to buy the brand-new SUV based on these features. using different classifications model such as SVM, Naïve Bays, and KNN

 

Feature Scaling

Before training the classifiers, the dataset was preprocessed. The features (age and estimated salary) were extracted from the dataset, and the target variable (purchase) was separated. The dataset was then split into training and testing sets using a 75:25 ratio.

 








Classification

Three classification algorithms were employed in this project: SVM, Naïve Bayes, and KNN.

Support Vector Machine (SVM):

SVM is a powerful classification algorithm that finds an optimal hyperplane to separate data points belonging to different classes. In this project, the SVM classifier was trained using the training set. The Radial Basis Function (RBF) kernel was used, and the classifier's performance was evaluated on the testing set.

 

Naïve Bayes

Naïve Bayes is a probabilistic classifier that applies Bayes' theorem with the assumption of independence between features. The Naïve Bayes classifier was trained using the training set and its performance was evaluated on the testing set.

 

K-Nearest Neighbors (KNN)

KNN is a simple yet effective classification algorithm that assigns a class label to a data point based on the majority vote of its k nearest neighbors. The KNN classifier was trained using the training set, with k set to 5 and the Minkowski distance metric with p=2 (Euclidean distance). The classifier's performance was evaluated on the testing set.

 








Result and Discussion

After training the classifiers, the performance of each algorithm was evaluated based on their accuracy in predicting the target variable on the testing set. The accuracy metric provides a measure of how well the classifiers can correctly classify new data points. The predictions were made on a new data point [30, 87000], and the KNN classifier correctly predicted that the user did not make a purchase.

It is worth mentioning that the choice of the number of neighbors (k) in the KNN classifier and the selection of the kernel function in the SVM classifier can have an impact on the classification performance. Further experimentation and fine-tuning of these parameters could potentially improve the accuracy of the classifiers.

Conclusion

	In conclusion, this data mining project successfully applied SVM, Naïve Bayes, and KNN algorithms for classification. The project involved loading and preprocessing the dataset, splitting it into training and testing sets, applying feature scaling, training the classifiers, and making predictions on new data points. The KNN classifier achieved the highest accuracy, followed by the SVM and Naïve Bayes classifiers. These results provide valuable insights into the purchase behavior of social network users and can be used for targeted marketing strategies or other business applications. Further research could explore additional algorithms and parameter tuning to improve the classification performance.



Step2

Replacing Some Values in The Dataset With NaN

Some values in the dataset were replaced with NaN to simulate missing data. The result of this replacement is as follows:

 


Step3

Replacing Nan Values With The Mean Of The Corresponding Column

The NaN values obtained in the previous step were replaced with the mean value of the corresponding column. The result of this replacement is as follows:

 



Step4

Classification On The New Dataset Using Support Vector Machine (SVM), 
Naïve Bayes, And K-Nearest Neighbors (KNN) Algorithms

Most of the steps outlined above have been repeated in this current step, as evident in the accompanying code file.

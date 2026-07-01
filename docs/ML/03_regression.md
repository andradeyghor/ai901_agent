Regression models are trained to predict numeric label values based on training data that includes both features and known labels. The process for training a regression model (or indeed, any supervised machine learning model) involves multiple iterations in which you use an appropriate algorithm (usually with some parameterized settings) to train a model, evaluate the model's predictive performance, and refine the model by repeating the training process with different algorithms and parameters until you achieve an acceptable level of predictive accuracy.

Four key elements of the training process for supervised machine learning models:

1. Split the training data (randomly) to create a dataset with which to train the model while holding back a subset of the data that you'll use to validate the trained model.
2. Use an algorithm to fit the training data to a model. In the case of a regression model, use a regression algorithm such as linear regression.
3. Use the validation data you held back to test the model by predicting labels for the features.
4. Compare the known actual labels in the validation dataset to the labels that the model predicted. Then aggregate the differences between the predicted and actual label values to calculate a metric that indicates how accurately the model predicted for the validation data.

After each train, validate, and evaluate iteration, you can repeat the process with different algorithms and parameters until an acceptable evaluation metric is achieved.


## Training a regression model
We'll start by splitting the data and using a subset of it to train a model. 
To get an insight of how these x and y values might relate to one another, we can plot them as coordinates along two axes.

Now we're ready to apply an algorithm to our training data and fit it to a function that applies an operation to x to calculate y. One such algorithm is linear regression, which works by deriving a function that produces a straight line through the intersections of the x and y values while minimizing the average distance between the line and the plotted points.

The line is a visual representation of the function in which the slope of the line describes how to calculate the value of y for a given value of x.


## Evaluating a regression model
To validate the model and evaluate how well it predicts, we held back some data for which we know the label (y) value.

We can use the model to predict the label for each of the observations in this dataset based on the feature (x) value; and then compare the predicted label (ŷ) to the known actual label value (y).

We can plot both the predicted and actual labels against the feature values.

The predicted labels are calculated by the model so they're on the function line, but there's some variance between the ŷ values calculated by the function and the actual y values from the validation dataset; which is indicated on the plot as a line between the ŷ and y values that shows how far off the prediction was from the actual value.

Based on the differences between the predicted and actual values, you can calculate some common metrics that are used to evaluate a regression model.


## Mean Absolute Error (MAE)
Mean Absolute Error (MAE) is a statistical metric used to measure the average magnitude of errors between predicted and actual values. It calculates how far off predictions are from reality, on average, and is expressed in the same units as the original data.


## Mean Squared Error (MSE)
The mean absolute error metric takes all discrepancies between predicted and actual labels into account equally. However, it may be more desirable to have a model that is consistently wrong by a small amount than one that makes fewer, but larger errors. One way to produce a metric that "amplifies" larger errors by squaring the individual errors and calculating the mean of the squared values. This metric is known as the mean squared error (MSE).


## Root Mean Squared Error (RMSE)
The mean squared error helps take the magnitude of errors into account, but because it squares the error values, the resulting metric no longer represents the quantity measured by the label.
If we want to measure the error in terms of the quantity measured by the label, we need to calculate the square root of the MSE; which produces a metric called, unsurprisingly, Root Mean Squared Error. 


## Coefficient of determination (R2)
In a linear regression model, the training algorithm fits a straight line that minimizes the mean variance between the function and the known label values. The coefficient of determination (more commonly referred to as R2 or R-Squared) is a metric that measures the proportion of variance in the validation results that can be explained by the model, as opposed to some anomalous aspect of the validation data (for example, a day with a highly unusual number of ice creams sales because of a local festival).

The calculation for R2 is more complex than for the previous metrics. It compares the sum of squared differences between predicted and actual labels with the sum of squared differences between the actual label values and the mean of actual label values, like this:

R2 = 1- ∑(y-ŷ)2 ÷ ∑(y-ȳ)2

The important point is that the result is a value between 0 and 1 that describes the proportion of variance explained by the model. In simple terms, the closer to 1 this value is, the better the model is fitting the validation data.


## Iterative training
The metrics described above are commonly used to evaluate a regression model. In most real-world scenarios, a data scientist will use an iterative process to repeatedly train and evaluate a model, varying:
* Feature selection and preparation (choosing which features to include in the model, and calculations applied to them to help ensure a better fit).
* Algorithm selection (We explored linear regression in the previous example, but there are many other regression algorithms)
* Algorithm parameters (numeric settings to control algorithm behavior, more accurately called hyperparameters to differentiate them from the x and y parameters).

After multiple iterations, the model that results in the best evaluation metric that's acceptable for the specific scenario is selected.
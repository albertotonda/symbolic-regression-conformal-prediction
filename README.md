# Symbolic Regression for Conformal Prediction Confidence Intervals

## Ideas
Use the boundaries of the Mondrian predictors as initial seeds for the values in the GP.

## Notes

### 2024-06-04
We should probably go for a RandomForest with 500 trees.

### 2024-06-04
physiochemical_protein with high settings for PySR makes the program crash. Probably need some re-runs with lower settings.

### 2024-06-03
There are some issues that previously did not exist with KNN. Maybe it's due to scikit-learn version 1.5; I could try to create an environment with scikit-learn==1.4.2; No, actually it turns out that we just need to update threadpoolctl to the latest version.

### 2024-05-14
Easier approach to fitness: just fit the distance between predicted and true point. However, penalize more if the distance is smaller.

### 2024-05-13
What we would like to do:
1. minimize number of measured points outside of [predicted-ci, predicted+ci]
2. minimize amplitude of confidence intervals

Fitness function: predict value of the confidence interval; apply to predicted point: 
1. if measured point falls within confidence interval, -1 on first fitness (SATURATE ON COVERAGE REQUIRED)
2. if measured point is out, minimize delta out
3. then, minimize size of confidence intervals

## Resources
Libraries for conformal prediction: https://github.com/henrikbostrom/crepes
Examples: https://crepes.readthedocs.io/en/latest/crepes_nb_wrap.html#Investigating-the-prediction-intervals
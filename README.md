# Symbolic Regression for Conformal Prediction Confidence Intervals

## Notes

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
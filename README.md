# Process model forecasting
Process Model Forecasting Using Time Series Analysis of Event Sequence Data

## About
Process analytics is the field focusing on predictions for individual process instances or overall process models. At the instance level, various novel techniques have been recently devised, tackling next activity, remaining time, and outcome prediction. At the model level, there is a notable void. It is the ambition of this paper to fill this gap. To this end, we develop a technique to forecast the entire process model from historical event data. A forecasted model is a will-be process model representing a probable future state of the overall process. Such a forecast helps to investigate the consequences of drift and emerging bottlenecks. 
Our technique builds on a representation of event data as multiple time series, each capturing the evolution of a behavioural aspect of the process model, such that corresponding forecasting techniques can be applied.
Our implementation demonstrates the accuracy of our technique on real-world event log data.

## Implementation
The implementation is based on Python and uses the following packages:
- [pm4py](https://pm4py.fit.fraunhofer.de/)
- [statsmodels](https://www.statsmodels.org/stable/index.html)
- [scikit-learn](https://scikit-learn.org/stable/)
- [numpy](https://numpy.org/)
- [arch package](https://pypi.org/project/arch/)
- [entropia](https://github.com/jbpt/codebase/tree/master/jbpt-pm/entropia): make sure to include jbpt-pm-entropia-1.6.jar as well as the lib directory next to the Python code as this is used to calculate the entropic relevance

## Parameters and use
You can use the main Python file create_dfg_forecasts.py with the following parameters:

- dataset (e.g. 'rtfmp'): should be the name of an included .xes file
- agg_type ('equisize' or 'equitemp'): aggregation used to collect events in intervals (equisize: the same number of events in every intervals, equisize: every intervals has the same timespan)
- no_pairs (integer)
- horizon (integer): number of intervals forecasted, i.e., the forecasting horizon
- no_intervals_all (integer): number of intervals in whic the log will be split
- no_intervals (integer): number of intervals (out of all the intervals) used for the whole dataset (length of training set = no_intervals - horizon)
- no_folds (integer): number of folds used for cross-validation

The generated outcome (.csv file) will include the cosine distance, root mean square error and entropic relevance of both the forecasted and actual DFGs.

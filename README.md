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

## Datasets
We used 6 datasets, as-is from the following locations:
- [RTFMP](https://doi.org/10.4121/uuid:270fd440-1057-4fb9-89a9-b699b47990f5)
- [BPI12](https://doi.org/10.4121/uuid:3926db30-f712-4394-aebc-75976070e91f)
- [BPI17](https://doi.org/10.4121/uuid:5f3067df-f10b-45da-b98b-86ae4c7a310b)
- [BPI18](https://doi.org/10.4121/uuid:3301445f-95e8-4ff0-98a4-901f1f204972)
- [Sepsis](https://doi.org/10.4121/uuid:915d2bfb-7e84-49ad-a286-dc35f063a460)
- [Italian](https://doi.org/10.4121/uuid:0c60edf1-6f83-4e75-9367-4c63b3e9d5bb)

Note that the ones included in [datasets](/datasets) need to be unzipped.

## Parameters and use
You can use the main Python file create_dfg_forecasts.py with the following parameters:

- dataset (e.g. 'rtfmp'): should be the name of an included .xes file
- agg_type ('equisize' or 'equitemp'): aggregation used to collect events in intervals (equisize: the same number of events in every intervals, equisize: every intervals has the same timespan)
- no_pairs (integer): use to limit to the #no_pairs most frequent DF pairs. Use 0 to use all activity pairs.
- horizon (integer): number of intervals forecasted, i.e., the forecasting horizon
- no_intervals_all (integer): number of intervals in whic the log will be split
- no_intervals (integer): number of intervals (out of all the intervals) used for the whole dataset (length of training set = no_intervals - horizon)
- no_folds (integer): number of folds used for cross-validation

The generated outcome (.csv file) will include the cosine distance, root mean square error and entropic relevance of both the forecasted and actual DFGs.

import pm4py
import json
import os
from pm4py.objects.log.importer.xes import importer as xes_importer
import numpy as np
from math import sqrt

from sklearn.metrics import mean_squared_error, precision_score, recall_score, mean_absolute_error
from scipy.spatial import distance

from import_data import read_data_equisize, read_data_equitemp, determine_cutoff_point, ActivityPair
from forecasting import ARf, ARIMAf, SESf, HWf, NAVf, ESf, GARCHf
from operations import calculate_entropic_relevance

import networkx as nx
import warnings
warnings.filterwarnings("ignore")


############
# Parameters

dataset = 'italian'
agg_type = 'equisize'
no_pairs = 0
horizon = 25
no_intervals = 75
no_folds = 10
no_intervals_all = 100

# Parameters
############


variant = xes_importer.Variants.ITERPARSE
paras = {variant.value.Parameters.MAX_TRACES: 1000000000}
log = xes_importer.apply(dataset + '.xes', parameters=paras)

# read and encode data
activity_names = pm4py.get_attribute_values(log, 'concept:name')
no_act = len(activity_names)
act_map = {}
reverse_map = {}
for a, value in enumerate(activity_names.keys()):
    act_map[value] = a
    reverse_map[a] = value

# add start and end points for DFGs
act_map['start'] = no_act
act_map['end'] = no_act + 1
reverse_map[no_act] = 'start'
reverse_map[no_act+1] = 'end'
no_act += 2
print('Activity encoding:', act_map)

# store all directly-follows occurrences as activity pairs
apairs = []
for t, trace in enumerate(log):
    for e, event in enumerate(trace):
        if e == len(trace) - 1:
            continue
        ap = ActivityPair(event['concept:name'], trace[e+1]['concept:name'], trace[e+1]['time:timestamp'], event, trace[e+1], t)
        apairs.append(ap)

sorted_aps = sorted(apairs)
print('#DFs:', len(sorted_aps))

interval_width = int(len(sorted_aps) / no_intervals_all)

# transform the separate DFs into a matrix dfg_time_matrix_org of
# interval x activities x activities
if os.path.isfile(f'dfg___time_matrix_{dataset}_{agg_type}.npy'):
    with open(f'dfg_time_matrix_{dataset}_{agg_type}.npy', 'rb') as f:
        dfg_time_matrix_org = np.load(f)
    print('Read matrix:', dfg_time_matrix_org.shape)
else:
    print('Creating DFG matrix')
    if agg_type == 'equisize':
        dfg_time_matrix_org, interval_timings = read_data_equisize(no_intervals_all, interval_width, sorted_aps,
                                                                  act_map, log, dataset)
    else:
        dfg_time_matrix_org, interval_timings = read_data_equitemp(no_intervals_all, interval_width, sorted_aps, act_map, log, dataset)

    with open(f'dfg_time_matrix_{dataset}_{agg_type}.npy', 'wb') as f:
        np.save(f, dfg_time_matrix_org)

# determine cutoff in case only the most frequent subset of DF pairs needs to be retained
if no_pairs == 0:
    cutoff = 0
else:
    cutoff = determine_cutoff_point(act_map, dfg_time_matrix_org, no_pairs)

# reduce matrix according to parameter settings
dfg_time_matrix = dfg_time_matrix_org[:no_intervals, ::, ::]

techniques = ['nav', 'ar1', 'ar2', 'ar4', 'arima211', 'arima212', 'hw', 'garch']

dfg_result_matrix = {}
dfg_actual_matrix = np.zeros([no_act, no_act, no_folds, horizon], dtype=int)

for technique in techniques:
    dfg_result_matrix[technique] = np.zeros([no_act, no_act, no_folds, horizon], dtype=float)

chosen_pairs = set()
# forecast DFs of all activity pairs
for act, a in act_map.items():
    for act_2, a2 in act_map.items():

        # by default only time series with at least 1 DF will be selected
        if np.sum(dfg_time_matrix[:, a, a2]) > cutoff:
            print('DFG', act, 'and', act_2)
            chosen_pairs.add((a, a2))

            # get DF
            array = dfg_time_matrix[:, a, a2]

            techniques = dict()
            techniques['nav'] = NAVf()
            # techniques['ar1'] = ARf(1)
            techniques['ar2'] = ARf(2)
            techniques['ar4'] = ARf(4)
            # techniques['arima211'] = ARIMAf(2, 1, 1)
            techniques['arima212'] = ARIMAf(2, 1, 2)
            techniques['hw'] = HWf()
            techniques['garch'] = GARCHf()

            # cross-validation is applied
            for fold in range(0, no_folds):
                # offset for cross-validation
                offset = - fold - horizon
                x = array[:offset]

                if fold == 0:
                    y = array[offset:]
                else:
                    y = array[offset:(offset + horizon)]

                # store actual
                dfg_actual_matrix[a, a2, fold] = y

                for technique, implement in techniques.items():
                    y_pred = []
                    my_x = np.copy(x)

                    try:
                        # predict horizon steps ahead
                        y_hat = implement.fit(my_x, horizon)
                        y_pred = y_hat
                        dfg_result_matrix[technique][a, a2, fold] = y_pred
                    except:
                        dfg_result_matrix[technique][a, a2, fold] = np.full((horizon, ), 100000000)
            # End results of fold
        # End of -if > cutoff
    # End of act 2
# End of act 1

# Results per horizon/fold
with open(f'results_{dataset}_nopairs_{no_pairs}_nointervals_{str(no_intervals)}_{agg_type}.csv', 'w') as technique_fold_results:
    technique_fold_results.write('intervals,technique,fold,horizon,cosine,rmse,er_pred,er_actual\n')

    for technique in techniques:
        print(f'Technique {technique}')
        dfg_result_matrix_ar = dfg_result_matrix[technique]
        results_selected = np.zeros((len(chosen_pairs), no_folds, horizon))
        actual_selected = np.zeros((len(chosen_pairs), no_folds, horizon))

        for p, pair in enumerate(chosen_pairs):
            results_selected[p] = dfg_result_matrix_ar[pair[0], pair[1], ::, ::]
            actual_selected[p] = dfg_actual_matrix[pair[0], pair[1], ::, ::]

        for h in range(0, horizon):
            for fold in range(0, no_folds):
                # print(f'Fold {fold} - horizon {h}')

                #####################
                # Forecast DFG output
                nodes = []
                for act_name, act_code in act_map.items():
                    out_freq = int(np.sum(dfg_result_matrix_ar[act_code, ::, fold, h]))
                    in_freq = int(np.sum(dfg_result_matrix_ar[::, act_code, fold, h]))

                    if act_name == 'start':
                        nodes.append({'label': str(act_code), 'freq': out_freq, 'id': act_code})
                    else:
                        if in_freq > 0:
                            nodes.append({'label': str(act_code), 'freq': in_freq, 'id': act_code})

                arcs = []
                for a in range(0, len(act_map)):
                    for a2 in range(0, len(act_map)):
                        if int(dfg_result_matrix_ar[a, a2, fold, h]) > 0:
                            arcs.append({'from': a, 'to': a2, 'freq': int(dfg_result_matrix_ar[a, a2, fold, h])})

                # calculate entropic relevance
                dfg_file = {'nodes': nodes, 'arcs': arcs}
                r = json.dumps(dfg_file, indent=1)
                with open(f'temp.json', 'w') as dfg_write_file:
                    dfg_write_file.write(r)

                offset = no_intervals - horizon + h - fold
                er_technique = calculate_entropic_relevance(f'./logs/{dataset}_log_interval_{offset}-{no_intervals_all}_{agg_type}', 'temp')


                ###################
                # Actual DFG output
                nodes = []
                for act_name, act_code in act_map.items():
                    out_freq = int(np.sum(dfg_actual_matrix[act_code, ::, fold, h]))
                    in_freq = int(np.sum(dfg_actual_matrix[::, act_code, fold, h]))

                    if act_name == 'start':
                        nodes.append({'label': str(act_code), 'freq': out_freq, 'id': act_code})
                    else:
                        if in_freq > 0:
                            nodes.append({'label': str(act_code), 'freq': in_freq, 'id': act_code})

                arcs = []
                for a in range(0, len(act_map)):
                    for a2 in range(0, len(act_map)):
                        if int(dfg_actual_matrix[a, a2, fold, h]) > 0:
                            arcs.append({'from': a, 'to': a2, 'freq': int(dfg_actual_matrix[a, a2, fold, h])})

                # calculate entropic relevance
                dfg_file = {'nodes': nodes, 'arcs': arcs}
                r = json.dumps(dfg_file, indent=1)
                with open(f'temp.json', 'w') as dfg_write_file:
                    dfg_write_file.write(r)

                offset = no_intervals - horizon + h - fold
                er_actual = calculate_entropic_relevance(f'./logs/{dataset}_log_interval_{offset}-{no_intervals_all}_{agg_type}', 'temp')

                # store results
                results = np.reshape(results_selected[::, fold, h], (1, len(chosen_pairs)))
                actuals = np.reshape(actual_selected[::, fold, h], (1, len(chosen_pairs)))

                cosine = distance.cosine(results, actuals)
                rmse = sqrt(mean_squared_error(actuals, results))

                technique_fold_results.write(f'{no_intervals},{technique},{str(fold)},{str(h)},{cosine},{rmse},{er_technique},{er_actual}\n')

import numpy as np
import pm4py
import pytz
from datetime import datetime

from pm4py.algo.discovery.dfg.variants import native as dfg_factory
from collections import Counter
from pm4py.algo.filtering.log.timestamp import timestamp_filter
from pm4py.algo.discovery.dfg.variants import native

from pm4py.objects.dfg.utils.dfg_utils import infer_start_activities, infer_end_activities
from pm4py.objects.log.log import EventLog, Trace, Event
from pm4py.objects.log.exporter.xes.variants import etree_xes_exp as exporter


class ActivityPair:

    def __init__(self, a1, a2, timestamp, event, event2, trace_no):
        self.a1 = a1
        self.a2 = a2
        self.timestamp = timestamp
        self.event = event
        self.event2 = event2
        self.trace_no = trace_no

    def __str__(self):
        return self.a1 + ' before ' + self.a2 + ' at ' + str(self.timestamp)

    def __gt__(self, other):
        if self.timestamp > other.timestamp:
            return True
        else:
            return False


def read_data_equisize(no_intervals, interval_width, sorted_aps, act_map, log, dataset):
    no_act = len(act_map.keys())

    dfg_time_matrix = np.zeros([no_intervals, no_act, no_act], dtype=int)

    interval_timing = []

    no_events_sums = 0
    no_events_logs = 0
    no_dfs = 0

    shadow_dict = {}
    shadow_event_l = 0
    logs_finished = 0
    log_progression = {}
    for i in range(0, no_intervals):
        print('Interval ', i+1, '/', no_intervals)
        lower_bound = i * interval_width
        upper_bound = (i+1) * interval_width
        if i == (no_intervals - 1):
            upper_bound = len(sorted_aps)

        dfs = sorted_aps[lower_bound:upper_bound]
        no_dfs += len(dfs)

        print('#DFS:', len(dfs))

        empty_mat = np.zeros([no_act, no_act], dtype=float)

        # For output
        filtered_events = {}
        start = Event()
        end = Event()
        start['concept:name'] = str(act_map['start'])
        end['concept:name'] = str(act_map['end'])
        highest = datetime(1970, 1, 1, tzinfo=pytz.UTC)
        lowest = datetime(2050, 1, 1, tzinfo=pytz.UTC)

        log_dfs = {}
        for df in dfs:
            if df.trace_no not in log_dfs.keys():
                log_dfs[df.trace_no] = []
            log_dfs[df.trace_no].append(df)

        for trace_no, dfss in log_dfs.items():
            # print('\nTrace:', trace_no)
            sorted_dfs = sorted(dfss)
            filtered_events[trace_no] = []
            for df in sorted_dfs:
                # print(df)
                filtered_events[trace_no].append(df.event)
                no_events_sums += 1
            filtered_events[trace_no].append(sorted_dfs[len(sorted_dfs) - 1].event2)

            no_events_sums += 1

        for trace_no, events in filtered_events.items():
            empty_mat[act_map['start'], act_map[events[0]['concept:name']]] += 1
            empty_mat[act_map[events[-1]['concept:name']], act_map['end']] += 1

        # Export filtered events to interval event logs
        new_log = EventLog()
        no_eve = 0
        for t, trace in enumerate(log):
            if i < 50:
                if t not in shadow_dict.keys():
                    shadow_dict[t] = []
            new_trace = Trace()
            for trace_no, events in filtered_events.items():
                if t == trace_no:
                    for event in trace:
                        if event in events:
                            if event['time:timestamp'] < lowest:
                                lowest = event['time:timestamp']
                            if event['time:timestamp'] > highest:
                                highest = event['time:timestamp']
                            new_event = Event()
                            new_event['concept:name'] = str(act_map[event['concept:name']])
                            new_trace.append(new_event)
                            no_events_logs += 1
                            no_eve += 1
                            if i < 50:
                                shadow_dict[t].append(new_event)
                                shadow_event_l += 1
            if len(new_trace) > 0:
                # new_trace.append(end)
                new_log.append(new_trace)

        exporter.apply(new_log, './logs/' + dataset + '_log_interval_' + str(i) + '-'
                       + str(no_intervals) + '_equisize.xes')

        print('#Events:', no_eve)
        print('#Events log:', no_events_logs)
        print('#Events shadow low:', shadow_event_l)
        for act_pair in dfs:
            a1 = act_map[act_pair.a1]
            a2 = act_map[act_pair.a2]
            empty_mat[a1, a2] += 1

        dfg_time_matrix[i] = empty_mat

        interval_timing.append((lowest, highest))

    print('Event sums:', no_events_sums)
    print('Event logs:', no_events_logs)
    print('#DFS:', no_dfs)
    print('Logs finished:', logs_finished)
    return dfg_time_matrix, interval_timing


def read_data_equitemp(no_intervals, interval_width, sorted_aps, act_map, log, dataset):
    timestamps = pm4py.get_attribute_values(log, 'time:timestamp')
    print('Earliest:', min(timestamps))
    print('Latest:', max(timestamps))
    interval_length = (max(timestamps) - min(timestamps)) / no_intervals
    print('Interval length:', interval_length)

    no_act = len(act_map.keys())

    dfg_time_matrix = np.zeros([no_intervals, no_act, no_act], dtype=int)

    interval_timing = []
    no_events_sums = 0
    no_events_logs = 0
    no_dfs = 0
    for i in range(0, no_intervals):
        print('Interval ', i, '/', no_intervals)
        lower_bound = min(timestamps) + i * interval_length
        if i == (no_intervals - 1):
            upper_bound = min(timestamps) + (i + 1) * interval_length * 2
        else:
            upper_bound = min(timestamps) + (i + 1) * interval_length
        lb = lower_bound
        ub = upper_bound
        print(lb)
        print(ub)

        dfs = []
        empty_mat = np.zeros([no_act, no_act], dtype=float)

        filtered_events = {}
        start = Event()
        end = Event()
        start['concept:name'] = str(act_map['start'])
        end['concept:name'] = str(act_map['end'])
        highest = datetime(1970, 1, 1, tzinfo=pytz.UTC)
        lowest = datetime(2050, 1, 1, tzinfo=pytz.UTC)

        count = 0
        for df in sorted_aps:
            if ub > df.event2['time:timestamp'] >= lb:# and ub > df.event['time:timestamp'] >= lb:
                dfs.append(df)

        no_dfs += len(dfs)

        log_dfs = {}
        for df in dfs:
            if df.trace_no not in log_dfs.keys():
                log_dfs[df.trace_no] = []
            log_dfs[df.trace_no].append(df)

        for trace_no, dfss in log_dfs.items():
            # print('\nTrace:', trace_no)
            sorted_dfs = sorted(dfss)
            filtered_events[trace_no] = []
            for df in sorted_dfs:
                # print(df)
                filtered_events[trace_no].append(df.event)
                no_events_sums += 1
            filtered_events[trace_no].append(sorted_dfs[len(sorted_dfs)-1].event2)
            no_events_sums += 1

        print('#traces:', len(log_dfs))

        for trace_no, events in filtered_events.items():
            empty_mat[act_map['start'], act_map[events[0]['concept:name']]] += 1
            empty_mat[act_map[events[-1]['concept:name']], act_map['end']] += 1

        # Export filtered events to interval event logs
        new_log = EventLog()
        no_eve = 0
        for t, trace in enumerate(log):
            new_trace = Trace()
            # new_trace.append(start)
            for trace_no, events in filtered_events.items():
                if t == trace_no:
                    for event in trace:
                        if event in events:
                            if event['time:timestamp'] < lowest:
                                lowest = event['time:timestamp']
                            if event['time:timestamp'] > highest:
                                highest = event['time:timestamp']
                            new_event = Event()
                            new_event['concept:name'] = str(act_map[event['concept:name']])
                            new_trace.append(new_event)
                            no_events_sums += 1
                            no_eve += 1
            if len(new_trace) > 0:
                # new_trace.append(end)
                new_log.append(new_trace)
        exporter.apply(new_log, './logs/' + dataset + '_log_interval_' + str(i) + '-'
                       + str(no_intervals) + '_equitemp.xes')

        # print('no eve:', no_eve)
        for act_pair in dfs:
            a1 = act_map[act_pair.a1]
            a2 = act_map[act_pair.a2]
            empty_mat[a1, a2] += 1

        dfg_time_matrix[i] = empty_mat
        interval_timing.append((lowest, highest))
    print('Event sums:', no_events_sums)
    print('Event logs:', no_events_logs)
    print('#DFS:', no_dfs)

    return dfg_time_matrix, interval_timing


def determine_cutoff_point(act_map, dfg_time_matrix, no_pairs):
    counts = []
    for act, a in act_map.items():
        for act2, a2 in act_map.items():
            counts.append(np.sum(dfg_time_matrix[:, a, a2]))
    fc = Counter(counts)
    no_found = 0
    cutoff = 0
    for sum in sorted(fc.keys(), reverse=True):
        cutoff = sum
        no_found += fc[sum]
        if no_found > no_pairs:
            break
    print('CUTOFF:', cutoff, '#pairs:', no_found)

    return cutoff
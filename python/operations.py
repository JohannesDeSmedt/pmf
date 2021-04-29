import subprocess


def calculate_entropic_relevance(log_name, dfg_name):

    try:
        args = []
        args.append('java')
        args.append('-jar')
        args.append('jbpt-pm-entropia-1.6.jar')
        args.append('-r')
        args.append('-s')
        args.append('-rel')
        logstring = log_name + '.xes'
        args.append(logstring)
        predstring = dfg_name + '.json'
        args.append('-ret')
        args.append(predstring)

        # print(args)
        result = subprocess.check_output(args)
        result_s = float(result)
    except:
        result_s = 'NA'

    return result_s


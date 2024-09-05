from spyglass.shijiegu.load import load_run_sessions
from spyglass.shijiegu.Analysis_SGU import RippleTimesWithDecode, TrialChoice
import pandas as pd

def cont_vs_frag_occurrence_day(animal,dates_to_plot,encoding_set,classifier_param_name):
    (num_all,num_cont_all,
     num_frag_all,pct_cont_all,
     pct_frag_all,time_cont_all,time_frag_all) = ({},{},{},{},{},{},{})

    for d in dates_to_plot:
        nwb_copy_file_name = animal.lower() + d + '_.nwb'
        run_session_ids, run_session_names, pos_session_names = load_run_sessions(nwb_copy_file_name)

        (num_cont_day, num_frag_day,
         pct_cont_day, pct_frag_day,
         time_cont_day, time_frag_day) = ([],[],[],[],[],[])
        num_all_day = 0

        for ind in range(len(run_session_names)):
            session_name = run_session_names[ind]
            StateScript = pd.DataFrame(
                (TrialChoice & {'nwb_file_name':nwb_copy_file_name,'epoch_name':session_name}).fetch1('choice_reward'))

            trial_1_t = StateScript.loc[1].timestamp_O
            trial_last_t = StateScript.loc[len(StateScript)-1].timestamp_O
            session_duration = trial_last_t - trial_1_t

            key = {'nwb_file_name': nwb_copy_file_name,
                   'interval_list_name': session_name,
                   'classifier_param_name': classifier_param_name,
                   'encoding_set': encoding_set}
            ripple_times = pd.DataFrame((RippleTimesWithDecode & key
                             ).fetch1('ripple_times'))
            (num_cont,num_frag,
             pct_cont,pct_frag,
             time_cont,time_frag) = cont_vs_frag_occurrence(ripple_times)
            num_cont_day.append(num_cont/session_duration)
            num_frag_day.append(num_frag/session_duration)
            pct_cont_day.append(pct_cont)
            pct_frag_day.append(pct_frag)
            time_frag_day.append(time_frag/session_duration)
            time_cont_day.append(time_cont/session_duration)
            num_all_day = num_all_day + len(ripple_times)

        num_all[d] = num_all_day
        num_cont_all[d] = num_cont_day
        num_frag_all[d] = num_frag_day
        pct_cont_all[d] = pct_cont_day
        pct_frag_all[d] = pct_frag_day
        time_cont_all[d] = time_cont_day
        time_frag_all[d] = time_frag_day
    return num_all, num_cont_all, num_frag_all, pct_cont_all, pct_frag_all, time_cont_all, time_frag_all





def cont_vs_frag_occurrence(ripple_times):
    num_cont = find_SWR_number_involved(ripple_times,cont = True)
    num_frag = find_SWR_number_involved(ripple_times,cont = False)
    pct_cont = find_SWR_pct_involved(ripple_times,cont = True)
    pct_frag = find_SWR_pct_involved(ripple_times,cont = False)
    time_cont = find_SWR_time_sum(ripple_times,cont = True)
    time_frag = find_SWR_time_sum(ripple_times,cont = False)

    return (num_cont,num_frag,pct_cont,pct_frag,time_cont,time_frag)

def find_SWR_time_sum(ripple_times,cont = True):
    sum = 0

    for i in ripple_times.index:
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl
        for intvl in intvls:
            sum += intvl[1] - intvl[0]
    return sum

def find_SWR_number_involved(ripple_times,cont = True):

    sum = 0
    for i in ripple_times.index:
        involved = False
        if cont:
            intvls = ripple_times.loc[i].cont_intvl
        else:
            intvls = ripple_times.loc[i].frag_intvl

        for intvl in intvls:
            if (intvl[1] - intvl[0]) > 0.02:
                involved = True
                break
        if involved:
            sum += 1

    return sum

def find_SWR_pct_involved(ripple_times,cont = True):
    num = find_SWR_number_involved(ripple_times,cont = cont)
    return num/len(ripple_times)
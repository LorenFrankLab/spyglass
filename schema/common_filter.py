#code to define filters that can be applied to continuous time data
import datajoint as dj
import scipy.signal as signal

schema = dj.schema('common_filter')


@schema
class Filter(dj.Imported):
    definition = """                                                                             
    filter_name: varchar(80)    # descriptive name of this filter
    filter_samp_freq: int       # sampling frequency
    ---
    filter_comments: varchar(255)   # comments about the filter
    filter_numerator: blob          # numpy array containing the filter numerator (b)
    filter_denominator: blob        # numpy array containing filter denominator (a)                                                  
    """

    def zpk(self):
        # return the zeros, poles, and gain for the filter
        return signal.tf2zpk(filter_numerator, filter_denominator)
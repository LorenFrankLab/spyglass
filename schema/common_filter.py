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
    filter_left_stop=0: float         # highest frequency for stop band for high pass side of filter
    filter_left_pass=0: float         # lowest frequency for pass band of high pass side of filter
    filter_right_stop=0: float         # highest frequency for stop band for low pass side of filter
    filter_right_pass=0: float         # lowest frequency for pass band of low pass side of filter
    filter_b: blob                  # numpy array containing the filter numerator 
    filter_a: blob                  # numpy array containing filter denominator                                                   
    """

    def zpk(self):
        # return the zeros, poles, and gain for the filter
        return signal.tf2zpk(self.filter_b, self.filter_a)
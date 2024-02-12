from spyglass.shijiegu.load import load_epoch_data
from spyglass.shijiegu.Analysis_SGU import plot_decode_spiking
import numpy as np


class SessionData:
    def __init__(self, info={}, #session trial number etc
                 remote_time=None,
                 linear_position=None,
                 decode=None,
                 recordings=None,neural=None,mua=None,
                 ripple=None,
                 theta=None,
                 head_speed=None,head_orientation=None,ripple_consensus=None,
                 posterior_arm=None): #data frame, speed
        self.info= info #can put anything here, such as nwb_copy_file_name,epoch_num,trial number, etc
        self.remote_time = remote_time #xr, binary, 1 indicate remote time bins
        self.linear_position=linear_position # animal's location,xr
        
        self.decode = decode #decode, xarray object, has its own time axis
        
        self.recordings = recordings #neural recording
        self.neural = neural
        self.mua=mua
        
        self.ripple = ripple
        
        self.ripple_consensus = ripple_consensus
        
        self.theta = theta
        
        self.head_speed=head_speed #df, has their own timestamps
        self.head_orientation=head_orientation #df, has their own timestamps
        self.posterior_arm=posterior_arm
        
    def load(self,nwb_copy_file_name,epoch_num):
        
        (epoch_name,log_df,decode,
         head_speed,head_orientation,linear_position,
         theta,ripple,
         neural,mua,recordings)=load_epoch_data(nwb_copy_file_name,epoch_num)
        
        self.info['nwb_copy_file_name']=nwb_copy_file_name
        self.info['epoch_num']=epoch_num
        self.remote_time = None
        self.linear_position=linear_position # animal's location
        
        self.decode = decode #decode, xarray object, has its own time axis
        
        self.recordings = recordings #neural recording
        self.neural = neural
        self.mua=mua
        
        self.ripple = ripple
        self.ripple_consensus = None
        
        self.theta = theta
        
        self.head_speed=head_speed #xr, has their own timestamps
        self.head_orientation=head_orientation #xr, has their own timestamps
        
    def plot(self,title='',savefolder=[],savename=[]): #plot the entire session
        linear_position = self.linear_position
        plottimes=[np.array(linear_position.time[0]),np.array(linear_position.time[-1])]
        
        time_slice = slice(plottimes[0], plottimes[1])
        remote_time=self.remote_time
        remote_time_d=np.array(remote_time.sel(time=time_slice).to_array()).ravel()
        remote_time_t=np.array(remote_time.sel(time=time_slice).time)
        t0t1_ind=find_start_end(remote_time_d)
        t0t1=np.array(linear_position.time)[t0t1_ind]
        
        decode=self.decode
        theta=self.theta
        neural=self.neural
        mua=self.mua
        ripple=self.ripple
        head_speed=self.head_speed
        head_orientation=self.head_orientation
        ripple_consensus=self.ripple_consensus
        
        plot_decode_spiking(plottimes,t0t1,linear_position,decode,theta,
          neural,mua,ripple,head_speed,head_orientation,ripple_consensus,
          title=title,savefolder=savefolder,savename=savename,
              simple=False)
        
def find_start_end(binary_string):
    diff=np.diff(np.concatenate(([0],binary_string,[0])))
    op=np.argwhere(diff==1)
    ed=np.argwhere(diff==-1)-1
    return np.concatenate((op,ed),axis=1)
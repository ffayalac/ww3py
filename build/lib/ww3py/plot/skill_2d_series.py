import os,sys
PROJECT_ROOT = os.path.abspath(os.path.join(
                  os.path.dirname(__file__), 
                  os.pardir)
)
sys.path.append(PROJECT_ROOT)

import util
from . import init_custom

import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
import pandas as pd
import matplotlib.dates as mdates
import math
import numpy as np
import os
import datetime as dt
from scipy import interpolate
import datetime as dt


def preparing_bouy_data(spec_per_time):
        efth_min=1e-3 # Minimum value for wave energy
        spec_per_time=np.concatenate((spec_per_time,spec_per_time[:,0].reshape(-1,1)),axis=1)
        spec_per_time=np.where(spec_per_time>= efth_min, spec_per_time,efth_min)
        return spec_per_time

def preparing_model_data(spec_to_plot_model):
        spec_to_plot_model=np.concatenate((spec_to_plot_model[:,::-1][:,8:],spec_to_plot_model[:,::-1][:,:8]),axis=1) #0-360
        spec_to_plot_model=np.concatenate((spec_to_plot_model[:,:],spec_to_plot_model[:,0].reshape(-1,1)),axis=1) 
        return spec_to_plot_model

class Skill_2d_series():
        def __init__(self,root_path,ini_date,end_date,metrics_2d) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_2d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date                
                self.buoys_id = ['42057','42058','42059','42060']
                self.metrics_2d = metrics_2d
                self.dics_complete=np.arange(0,370,10)
        
        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.dics_ww3,self.spec_2d_model = util.read_data_spec_stations(f'{self.data_path}ww3.2020_spec_2d.nc')

                self.dics_data={}
                self.r_data={}
                self.theta_data={}

                for id in self.buoys_id:    
                        # Spectra from the buoy
                        self.time_buoy,self.freqs_buoy,self.dics_buoy,self.spec_2d_buoy=util.read_2d_spec_buoy(id)  # reading buoy results
                        self.idx_dates_buoy=np.where((self.time_buoy<=self.edate) & (self.time_buoy>=self.idate-relativedelta(hours=1)),True,False)
                        self.time_buoy=self.time_buoy[self.idx_dates_buoy]
                        self.spec_2d_buoy=self.spec_2d_buoy[self.idx_dates_buoy,:,:]

                        self.r,self.theta = np.meshgrid(self.freqs_buoy,np.radians(self.dics_complete))
                        self.spec_2d_buoy = np.array(list(map(preparing_bouy_data,self.spec_2d_buoy)))

                        self.new_x_index=(self.time_buoy+pd.Timedelta(minutes=20))

                        self.idx_dates_model=np.array(list(map(lambda x: True if x in self.new_x_index else False,self.time_ww3)))

                        # # Spectra from the model (ww3)
                        self.dics_complete_ww3=np.round((np.degrees(self.dics_ww3)+180)%360,1)
                        self.dic_complete_ww3_sorted=np.concatenate((self.dics_complete_ww3[::-1][8:],self.dics_complete_ww3[::-1][:8],np.array([360]))) # 0-360
                        self.r_ww3,self.theta_ww3 = np.meshgrid (self.freqs_ww3,np.radians(self.dic_complete_ww3_sorted))
        
                        self.spec_to_plot_model=self.spec_2d_model[id]
                        self.spec_to_plot_model=self.spec_to_plot_model[self.idx_dates_model,:,:]

                        self.spec_to_plot_model = np.array(list(map(preparing_model_data,self.spec_to_plot_model)))

                        self.sorted_dics_model=self.theta_ww3[:,0]
                        self.sorted_freqs_model=self.r_ww3[0,:]
                        self.sorted_energy = self.spec_to_plot_model

                        self.new_freqs=self.r[0,3:]
                        self.new_dics=self.sorted_dics_model
                        self.new_r,self.new_theta=np.meshgrid(self.new_freqs,self.new_dics)

                        self.metric_sum=np.empty((self.spec_to_plot_model.shape[0]))

                        for idx_time in range(self.spec_to_plot_model.shape[0]):
                                self.interp_function = interpolate.interp2d(self.sorted_dics_model[:-1], self.sorted_freqs_model, self.sorted_energy[idx_time,:,:-1], kind='cubic')  
                                # Creating the new freq and dirs arrays to interpolate
                                self.interp_energy=self.interp_function(self.new_dics[:-1],self.new_freqs)

                                max_norm=np.max(self.spec_2d_buoy[idx_time,:,:])

                                self.spec_buoy_smoothed=np.array(list(map(util.moving_average_filter,self.spec_2d_buoy[idx_time,:,:].T)))
                                self.spec_2d_buoy[idx_time,:,:]=self.spec_buoy_smoothed.T

                                self.spec_to_plot_buoy_nom=self.spec_2d_buoy[idx_time,:,:]/max_norm
                                self.interp_energy_nom=self.interp_energy/max_norm

                                self.norm_difference = self.spec_to_plot_buoy_nom[3:,:-1] - self.interp_energy_nom
                                self.norm_difference = np.concatenate((self.norm_difference[:,:],self.norm_difference[:,0].reshape(-1,1)),axis=1) 
                                self.metric_sum[idx_time] = np.sum(np.abs(self.norm_difference))

                        # Storaging metrics
                        self.dics_data[id]=dict(sum=pd.Series(self.metric_sum,index=self.new_x_index))
                
                        # print(self.plots_path[32:37],dt.datetime(2020,5,9,0,0,0),f'total error {id}',self.dics_data[id]['sum'][dt.datetime(2020,5,9,0,0,0)])

                return self.dics_data

        def plotting_one_skill_serie(self,axes,idx,key,dict_var,id_buoy,label,compare):
                try:
                        self.ax=axes[idx]
                except:
                        self.ax=axes
                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                self.fmt_day = mdates.DayLocator()

                self.dict_to_plot=dict_var[id_buoy]
                self.ax.plot(self.dict_to_plot[key],color='k')

                if compare == True:
                        self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                        where=self.dict_to_plot[key].values>=0, interpolate=True, color='tomato')
                        self.ax.fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                        where=self.dict_to_plot[key].values<=0, interpolate=True, color='darkcyan')
                        string1='$_{ctrl}$'
                        string2=f'$_{{{label}}}$'
                        self.ax.set(ylabel=f'{key}{string1} - {key}{string2}')

                else:
                        self.ax.set(ylabel=f'{key}')

                self.ax.grid(True,alpha=0.5)
                self.ax.axhline(y=0, color="black", linestyle="--")
                self.ax.xaxis.set_major_formatter(self.myFmt)
                self.ax.xaxis.set_minor_locator(self.fmt_day)

                return self.ax

        def setting_up_plot(self,label):
                self.dict=self.preparing_data()

                self.dict_axes={buoy:list(range(len(self.metrics_2d))) for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        # self.fig,self.ax=plt.subplots(len(self.metrics_1d),1,figsize=(12,int(len(self.metrics_1d)*2)))
                        self.fig,self.ax=plt.subplots(len(self.metrics_2d),1,figsize=(12,int(len(self.metrics_2d)*2)))

                        # Loop over each type of result (buoy or simulations)
                        self.dict_axes[id_buoy][0]=self.plotting_one_skill_serie(self.ax,0,'sum',self.dict,id_buoy,label,False)
                        self.fig.suptitle(f'Metrics for buoy {id_buoy} [{label}]',y=0.85)
                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}skill_spec_2d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

        def compare_another_conf(self,obj2,label):
                self.dict_var1=self.preparing_data()
                self.dict_var2=obj2.preparing_data()

                self.dict_axes={buoy:list(range(len(self.metrics_2d))) for buoy in self.buoys_id}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(len(self.metrics_2d),1,figsize=(12,int(len(self.metrics_2d)*2)))

                        # Loop over each type of result (buoy or simulations)
                        for idx,key in enumerate(self.metrics_2d):
                                self.dict_var2[id_buoy][key]=self.dict_var1[id_buoy][key]-self.dict_var2[id_buoy][key]
                                self.dict_axes[id_buoy][idx]=self.plotting_one_skill_serie(self.ax,idx,key,self.dict_var2,
                                                                                        id_buoy,label,True)  
                                self.fig.suptitle(f'Difference of metrics [ctrl-{label}] for buoy {id_buoy}',y=1.0)        

                                self.fig.savefig(f'{obj2.plots_path}skill_spec_2d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)


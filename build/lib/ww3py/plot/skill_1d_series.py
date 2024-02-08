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
import matplotlib.colors as colors
import datetime as dt


class Skill_1d_series():
        def __init__(self,root_path,ini_date,end_date,metrics_1d) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_1d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date                
                self.buoys_id = ['42057','42058','42059','42060']
                self.metrics_1d = metrics_1d
                self.type_lines=dict(RMSE='-',MBAE='-',corr='-')


        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.spec_1d_model = util.read_data_spec_1d_stations(f'{self.data_path}ww3.2020_spec_1d.nc')

                self.dics_data={}

                for id in self.buoys_id:    
                        # Spectra from the buoy
                        self.time_buoy,self.freqs_buoy,self.spec_1d_buoy=util.read_1d_spec_buoy(id)  # reading buoy results
                        self.idx_dates_buoy=np.where((self.time_buoy<=self.edate+relativedelta(hours=1)) & (self.time_buoy>=self.idate),True,False)
                        self.time_buoy=self.time_buoy[self.idx_dates_buoy]
                        self.spec_1d_buoy=self.spec_1d_buoy[self.idx_dates_buoy,3:]

                        # Spectral from model
                        self.new_x_index=(self.time_buoy+pd.Timedelta(minutes=20))
                        self.idx_dates_model=np.array(list(map(lambda x: True if x in self.new_x_index else False,self.time_ww3)))
                        self.spec_1d_model_id=self.spec_1d_model[id]
                        self.spec_1d_model_id=self.spec_1d_model_id[self.idx_dates_model,:]

                        # Smoothing 1D spectra
                        self.spec_1d_buoy=np.array(list(map(util.moving_average_filter,self.spec_1d_buoy)))

                        # Interpolating 1D spectra
                        self.spec_1d_model_id=np.array(list(map(util.interp_1d_spectra,np.tile(self.freqs_ww3,(self.spec_1d_model_id.shape[0],1)),\
                                                self.spec_1d_model_id,np.tile(self.freqs_buoy[3:],(self.spec_1d_model_id.shape[0],1)))))

                        # Computing metrics
                        metrics=np.array(list(map(util.metrics,self.spec_1d_buoy,self.spec_1d_model_id)))

                        self.dics_data[id]=dict(RMSE=pd.Series(metrics[:,0],index=self.new_x_index[:-1]),
                                                MBAE=pd.Series(metrics[:,1],index=self.new_x_index[:-1]),
                                                corr=pd.Series(metrics[:,2],index=self.new_x_index[:-1]))
                
                return self.dics_data

        def plotting_one_skill_serie(self,ax,idx,key,dict_var,id_buoy,label,compare):
                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                self.fmt_day = mdates.DayLocator()

                self.dict_to_plot=dict_var[id_buoy]
                ax[idx].plot(self.dict_to_plot[key],color='k',ls=self.type_lines[key])

                if compare == True:
                        if key=='corr':   
                                ax[idx].fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values>=0, interpolate=True, color='darkcyan')
                                ax[idx].fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values<=0, interpolate=True, color='tomato')
                                ax[idx].set(ylabel=f'{key} diff')

                        else:
                                ax[idx].fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values>=0, interpolate=True, color='tomato')
                                ax[idx].fill_between(self.dict_to_plot[key].index, self.dict_to_plot[key].values, 
                                                where=self.dict_to_plot[key].values<=0, interpolate=True, color='darkcyan')
                                string1='$_{ctrl}$'
                                string2=f'$_{{{label}}}$'
                                ax[idx].set(ylabel=f'{key}{string1} - {key}{string2} \n [$m^{2}s$]')


                else:
                        ax[idx].set(ylabel=f'{key} [$m^{2}s$]')

                ax[idx].grid(True,alpha=0.5)
                ax[idx].axhline(y=0, color="black", linestyle="--")
                ax[idx].xaxis.set_major_formatter(self.myFmt)
                ax[idx].xaxis.set_minor_locator(self.fmt_day)

                return ax

        def setting_up_plot(self,label,color):
                self.dict=self.preparing_data()
                print(label)

                self.dict_axes={buoy:list(range(len(self.metrics_1d))) for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(len(self.metrics_1d),1,figsize=(12,int(len(self.metrics_1d)*2)))

                        # Loop over each type of result (buoy or simulations)
                        for idx,key in enumerate(self.metrics_1d):
                                self.dict_axes[id_buoy][idx]=self.plotting_one_skill_serie(self.ax,idx,key,self.dict,
                                                                                       id_buoy,label,False)
                        self.fig.suptitle(f'Metrics for buoy {id_buoy} [{label}]',y=0.85)
                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}skill_spec_1d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

        def compare_another_conf(self,obj2,label,color):
                self.dict_var1=self.preparing_data()
                self.dict_var2=obj2.preparing_data()

                self.dict_axes={buoy:list(range(len(self.metrics_1d))) for buoy in self.buoys_id}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(len(self.metrics_1d),1,figsize=(12,int(len(self.metrics_1d)*2)))

                        # Loop over each type of result (buoy or simulations)
                        
                        for idx,key in enumerate(self.metrics_1d):
                                self.dict_var2[id_buoy][key]=self.dict_var1[id_buoy][key]-self.dict_var2[id_buoy][key]
                                self.dict_axes[id_buoy][idx]=self.plotting_one_skill_serie(self.ax,idx,key,self.dict_var2,
                                                                                 id_buoy,label,True)  
                        self.fig.suptitle(f'Difference of metrics [ctrl-{label}] for buoy {id_buoy}',y=0.95)        
                        # plt.subplots_adjust(wspace=0.4,hspace=0.6)                        

                        self.fig.savefig(f'{obj2.plots_path}skill_spec_1d_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)


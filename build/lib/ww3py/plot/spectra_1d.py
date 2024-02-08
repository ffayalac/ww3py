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
from scipy import interpolate
import matplotlib.colors as colors

class Spectra_1d():
        def __init__(self,root_path,ini_date,date) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/spectra_1d/'
                os.system(f'mkdir -p {self.plots_path}')
                self.date=date
                self.buoys_id = ['42057','42058','42059','42060']

        def preparing_data(self):
                # Reading model results
                self.time_ww3,self.freqs_ww3,self.spec_1d_model = util.read_data_spec_1d_stations(f'{self.data_path}ww3.2020_spec_1d.nc')

                self.dics_data={}
                self.freq_data={}

                for id in self.buoys_id:    
                        # Spectra from the buoy
                        self.date_buoy=self.date-pd.Timedelta(minutes=20)
                        self.time_buoy,self.freqs_buoy,self.spec_1d_buoy=util.read_1d_spec_buoy(id)  # reading buoy results
                        self.idx_inidate_buoy=self.time_buoy.get_loc(self.date_buoy)
                        self.spec_to_plot_buoy=self.spec_1d_buoy[self.idx_inidate_buoy,:]
                        self.spec_to_plot_buoy=util.moving_average_filter(self.spec_to_plot_buoy,3)

                        # Spectra from the model (ww3)
                        self.idx_dateini_ww3=self.time_ww3.get_loc(self.date)       
                        self.spec_to_plot_model=self.spec_1d_model[id]
                        self.spec_to_plot_model=self.spec_to_plot_model[self.idx_dateini_ww3,:]
                        
                        # Storaging spectra, freqs and dirs arrays in dictionaries
                        self.dics_data[id]=dict(buoy=self.spec_to_plot_buoy,model=self.spec_to_plot_model)
                        self.freq_data[id]=dict(buoy=self.freqs_buoy,model=self.freqs_ww3)
                
                return self.freq_data,self.dics_data

        def plotting_one_spectra(self,ax,freq,dict_var,id_buoy,type,label,color):
                
                self.spec=dict_var[id_buoy][type]
                if type=='buoy':
                        ax.plot(freq,self.spec,label=type,color='k',lw=1.5)
                else:
                        if label=='winp':
                                label='$cos^2$'
                        elif label=='bim2':
                                label='bim'
                        elif label =='all':
                                label='$cos^2$+bim'
                        ax.plot(freq,self.spec,label=label,color=color,lw=1.5)

                ax.grid(True,alpha=0.5)
                ax.set(ylabel="E(f) [$m^{2}s$]",xlabel='Frequency [Hz]',title=f'1D spectra - buoy {id_buoy} - {self.date}')
                ax.legend(ncol=2)

                return self.ax
        
        def setting_up_plot(self,label,color):
                self.freqs_to_plot,self.dict_to_plot=self.preparing_data()
                self.dict_axes={buoy:[1,2] for buoy in self.buoys_id}
                self.figs={}

                for id_buoy in self.buoys_id:
                        self.fig,self.ax=plt.subplots(1,1)

                        # Loop over each type of result (buoy or simulations)
                        for idx,key in enumerate(list(self.dict_to_plot[id_buoy].keys())):  
                                self.dict_axes[id_buoy][idx]=self.plotting_one_spectra(self.ax,self.freqs_to_plot[id_buoy][key],
                                                                                               self.dict_to_plot,id_buoy,key,label,color)
                        self.figs[id_buoy]=self.fig

                        
                        self.fig.savefig(f'{self.plots_path}spec_1d_{id_buoy}_{self.date.strftime("%d%m%y")}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

                return self.dict_axes,self.figs
        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.freqs_to_plot2,self.dict_to_plot2=obj2.preparing_data()
                self.dict_axes={buoy:[1,2] for buoy in self.buoys_id}

                for id_buoy in self.buoys_id:
                        for idx,key in enumerate(list(self.dict_to_plot2[id_buoy].keys())):  
                                if key!='buoy':
                                        self.dict_axes[id_buoy][idx]=self.plotting_one_spectra(dict2[id_buoy][idx],self.freqs_to_plot2[id_buoy][key],
                                                                                               self.dict_to_plot2,id_buoy,key,label,color)                        
                        figs2[id_buoy].savefig(f'{self.plots_path}spec_1d_{id_buoy}_{self.date.strftime("%d%m%y")}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
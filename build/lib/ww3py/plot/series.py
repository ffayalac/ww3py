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

class Series():
        def __init__(self,root_path,ini_date,end_date,vbles_to_plot) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/series/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = ['42057','42058','42059','42060']
                self.vbles_to_plot = vbles_to_plot

        def preparing_data(self):
                self.data_ounp=util.read_data_int_stations(f'{self.data_path}ww3.2020_tab_params.nc')
                self.locs_buoys = {'42057':(16.908,-81.422),'42058':(14.394,-74.816),'42059':(15.300,-67.483),
                                        '42060':(16.434,-63.329)}
                #time,freqs,data_spectra = util.read_data_src_stations(f'{self.run_path}ww3.2020_src_1d.nc','1d')
                #data_src_1d = data_spectra[id].sin[self.idate+relativedelta(hours=24):]

                self.hs={}
                self.u10={}
                self.dirs={}

                # Getting ERA5 series in the location of the buoys
                self.series_era5_buoys ={}
                for buoy in self.locs_buoys.keys():
                        self.lon=self.locs_buoys[buoy][1]+360
                        self.lat=self.locs_buoys[buoy][0]
                        self.result=util.read_era5_buoys(f'{self.run_path}/{self.idate.strftime("%Y%m%d")}_era5.nc',self.lon,self.lat)
                        self.series_era5_buoys[buoy]=self.result

                        self.data_buoy = util.ord_buoy_data(buoy)

                        # Hs data
                        self.hs_buoy=self.data_buoy.hs[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)] # First day is cutted off due to spin-up
                        self.new_x_index=(self.hs_buoy.index-pd.Timedelta(minutes=40))
                        self.hs_model=self.data_ounp[buoy].hs[self.new_x_index]
                        self.hs[buoy] = dict(model=self.hs_model,buoy=self.hs_buoy)        

                        # u10 data
                        self.wnd_spd_buoy=self.data_buoy.wspd[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wnd_spd_era = self.series_era5_buoys[buoy][self.idate+relativedelta(hours=24):]
                        self.u10[buoy] = dict(ERA5=self.wnd_spd_era,buoy=self.wnd_spd_buoy)

                        # wind and wave dir data
                        self.wvdir_buoy=self.data_buoy.dir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wndir_buoy=self.data_buoy.wndir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wvdir_model=self.data_ounp[buoy].dirp[self.new_x_index]
                        self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind=self.wndir_buoy)

                return  self.hs,self.u10,self.dirs

        def plotting_wave_integral(self,axes,idx,param,dict_var,id_buoy,flag,label,color):
                try:
                        self.ax=axes[idx]
                except:
                        self.ax=axes
                self.myFmt = mdates.DateFormatter('%m-%d-%y')
                self.fmt_day = mdates.DayLocator()
                self.colors_dict = dict(buoy='k',ERA5='darkcyan',wind='lightcoral')
                self.dict_vars=dict_var[param][id_buoy]
                for key in self.dict_vars.keys():
                        if 'model' in self.dict_vars.keys():
                                idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                                if len(idx_nans)>0:
                                        dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                        self.dict_vars['model']=self.dict_vars['model'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                        self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)

                                MSE = np.sqrt(((self.dict_vars['buoy'].values - self.dict_vars['model'].values)**2).mean()) 
                                RMSE = round(MSE,2)

                                bias = np.sqrt(((self.dict_vars['buoy'].values - np.mean(self.dict_vars['buoy'].values))**2).mean()) -\
                                        np.sqrt(((self.dict_vars['model'].values - np.mean(self.dict_vars['model'].values))**2).mean())
                                bias = round(bias,2)

                                MBE = np.mean(self.dict_vars['buoy'].values - self.dict_vars['model'].values)
                                MBE = round(MBE,2)

                        if flag == 'compare':
                                if label=='winp':
                                        label='$cos^2$'
                                elif label=='bim2':
                                        label='bim'
                                elif label =='all':
                                        label='$cos^2$+bim'
                                self.ax.plot(self.dict_vars['model'],c=color,label=f'{label} - RMSE: {RMSE} - MBE: {MBE}')
                                break
                        else:
                                if key =='model':
                                        if label == 'all':
                                                label='$cos^2$+bim'
                                        self.ax.plot(self.dict_vars[key],c=color,label=f'{label} - RMSE: {RMSE} - MBE: {MBE}')

                                else:
                                        self.ax.plot(self.dict_vars[key],c=self.colors_dict[key],label=key)

                if param =='hs':
                        self.hs_max=max([max([max(i) for i in list(bouy.values())]) for bouy in list(dict_var[param].values())])
                        # self.ax.set(ylabel="$H_{s}$" +" [m]",ylim=(0,4),yticks=range(0,4))
                        self.ax.set(ylabel="$H_{s}$" +" [m]",ylim=(0,self.hs_max+0.5),yticks=range(0,int(self.hs_max)+2))
                        self.ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.45),ncol=3, fancybox=True)
                elif param =='u10':
                        self.ax.set(ylabel="Wind speed [m/s]")
                elif param == 'dir':
                        self.ax.set(ylabel="Direction [°]")
                else:
                        self.ax.set(ylabel="Wave energy $[m^{2}]$")

                self.ax.xaxis.set_major_formatter(self.myFmt)
                self.ax.xaxis.set_minor_locator(self.fmt_day)
                self.ax.grid(True,linestyle='dotted')

                return self.ax

        def setting_up_plot(self,label,color):
                self.all_data_vbles=dict(hs=self.preparing_data()[0],u10=self.preparing_data()[1],dir=self.preparing_data()[2])
                self.dict_axes={buoy:list(range(len(self.vbles_to_plot))) for buoy in self.buoys_id}
                self.figs={buoy:list(range(len(self.vbles_to_plot))) for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.rows=int(len(self.vbles_to_plot))
                        if self.rows>=1:
                                self.fig,self.axes=plt.subplots(self.rows,1,figsize=(12,self.rows*2.5))
                                plt.suptitle(f'Time series for buoy {id_buoy}',y=1.30,fontsize=15)
                        else:
                                raise ValueError('The number of subplots is not expected')
                        
                        # Loop over each variable
                        for idx,el in enumerate(self.vbles_to_plot):  
                                self.dict_axes[id_buoy][idx]=self.plotting_wave_integral(self.axes,idx,el,self.all_data_vbles,id_buoy,'anything',label,color)
                        self.figs[id_buoy]=self.fig

                        self.fig.savefig(f'{self.plots_path}srs_{id_buoy}.png',dpi=400,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=dict(hs=obj2.preparing_data()[0],dir=obj2.preparing_data()[2])
                self.dict_axes={buoy:[1,2,3] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        # Loop over each variable
                        for idx,el in enumerate(self.vbles_to_plot):
                                if el != 'u10':
                                        self.dict_axes[id_buoy][idx]=self.plotting_wave_integral(dict2[id_buoy],idx,el,self.adding_variables,id_buoy,'compare',label,color)
                        
                        figs2[id_buoy].savefig(f'{self.plots_path}srs_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)

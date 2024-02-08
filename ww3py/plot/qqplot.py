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
import datetime as dt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error as mae
import string


def metrics(measures,model):

    MSE = np.sqrt(((measures - model)**2).mean()) 
    RMSE = round(MSE,2)

    bias = np.sqrt(((measures - np.mean(measures))**2).mean()) -\
            np.sqrt(((model - np.mean(model))**2).mean())
    bias = round(bias,2)

    MBE = np.mean(measures - model)
    MBE = round(MBE,2)

    corr, _ = pearsonr(measures,model)
    corr=round(corr,2)

    error = mae(measures,model)
    error=round(error,2)

    return RMSE,error,MBE,corr


def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique))

class QQplot():
        def __init__(self,root_path,ini_date,end_date,vbles_to_plot,buoys_id,locs_buoys) -> None:
                self.data_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/'
                self.run_path = f'{root_path}run/{ini_date.strftime("%Y%m%d%H")}/'
                self.plots_path = f'{root_path}data/{ini_date.strftime("%Y%m%d%H")}/plots/qqplots/'
                os.system(f'mkdir -p {self.plots_path}')
                self.idate=ini_date
                self.edate=end_date
                self.buoys_id = buoys_id
                self.locs_buoys = locs_buoys
                self.vbles_to_plot = vbles_to_plot

        def preparing_data(self):
                time_ERA5_hcst,hs_hindcast_ERA5_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs.nc')
                time_CSFR_hcst,hs_hindcast_CSFR_buoys=util.read_hindcast('/home/fayalacruz/ww3.202005_hs_CSFR.nc')
  
                self.data_ounp=util.read_data_int_stations(f'{self.data_path}ww3.{self.idate.year}_tab_params.nc')

                self.hs={}
                self.u10={}
                self.dirs={}

                self.wind_params=util.read_data_extra_stations(f'{self.data_path}ww3.{self.idate.year}_extra_params.nc')

                # Getting ERA5 series in the location of the buoys
                self.series_era5_buoys ={}
                self.series_era5_buoys_u ={}
                self.series_era5_buoys_v ={}

                for buoy in list(self.locs_buoys.keys()):
                        self.lon=self.locs_buoys[buoy][1]+360
                        self.lat=self.locs_buoys[buoy][0]

                        self.data_buoy = util.ord_buoy_data(buoy)

                        # Hs data
                        if self.idate.year == 2020:
                                self.result,self.result_u,self.result_v=util.read_era5_buoys(f'{self.run_path}/{self.idate.strftime("%Y%m%d")}_era5.nc',self.lon,self.lat)
                                self.series_era5_buoys[buoy]=self.result
                                self.series_era5_buoys_u[buoy]=self.result_u
                                self.series_era5_buoys_v[buoy]=self.result_v

                                self.hs_buoy=self.data_buoy.hs[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)] # First day is cutted off due to spin-up
                                self.new_x_index=(self.hs_buoy.index-pd.Timedelta(minutes=40))
                        else:
                                self.hs_buoy=self.data_buoy.hs[self.idate:self.edate] 
                                self.new_x_index=self.hs_buoy.index

                        self.hs_model=self.data_ounp[buoy].hs[self.new_x_index]
                        self.hs_hindcast_ERA5=pd.Series(hs_hindcast_ERA5_buoys[buoy],index=time_ERA5_hcst)
                        self.hs_hindcast_CSFR=pd.Series(hs_hindcast_CSFR_buoys[buoy],index=time_CSFR_hcst)
                        
                        self.hs[buoy] = dict(model=self.hs_model,buoy=self.hs_buoy,hindcast_ERA5=self.hs_hindcast_ERA5,
                                                hindcast_CSFR=self.hs_hindcast_CSFR)   

                        # u10 data
                        self.wnd_spd_buoy=self.data_buoy.wspd[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]

                        if self.idate.year == 2020:
                                self.wnd_spd_era = self.series_era5_buoys[buoy][self.idate+relativedelta(hours=24):]
                                self.u10[buoy] = dict(ERA5=self.wnd_spd_era,buoy=self.wnd_spd_buoy)
                                # self.u10[buoy] = dict(buoy=self.wnd_spd_buoy)
                        else:
                                self.u10[buoy] = dict(buoy=self.wnd_spd_buoy)

                        # wind and wave dir data
                        self.wvdir_buoy=self.data_buoy.dir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wndir_buoy=self.data_buoy.wndir[self.idate+relativedelta(hours=24):self.edate+relativedelta(hours=1)]
                        self.wvdir_model=self.data_ounp[buoy].dirp[self.new_x_index]
                        self.wndir_era5=(270-np.degrees(np.arctan2(self.series_era5_buoys_v[buoy][self.idate+relativedelta(hours=24):],self.series_era5_buoys_u[buoy][self.idate+relativedelta(hours=24):])))%360
                        self.wndir_model=self.wind_params[buoy].wnddir

                        # self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind_buoy=self.wndir_buoy,wind_era=self.wndir_era5)

                        self.dirs[buoy]=dict(model=self.wvdir_model,buoy=self.wvdir_buoy,wind_buoy=self.wndir_buoy)

                return  self.hs,self.u10,self.dirs

        def setting_up_plot(self,label,color):
                all_data_vbles=self.preparing_data()[0]
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                self.figs={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.fig,self.axes=plt.subplots(2,3,figsize=(10.5,7),constrained_layout=True,sharex=True,sharey=True)
                        
        #                 # Loop over each variable
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(self.axes,all_data_vbles,id_buoy,label,color)
                        self.fig.suptitle(f'Comparison of simulated and observed $H_s$ between ctrl experiment and other simulations - Buoy {id_buoy}')

                        self.figs[id_buoy]=self.fig
                        
                        self.fig.savefig(f'{self.plots_path}qqplot_{id_buoy}.png',dpi=400,bbox_inches='tight',pad_inches=0.05)
                return self.dict_axes,self.figs
        
        def plotting_one_scatter(self,axes,dict_var,id_buoy,label,color):
                self.dict_vars=dict_var[id_buoy]
                if 'model' in self.dict_vars.keys():
                        idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                        if len(idx_nans)>0:
                                dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                self.dict_vars['model']=self.dict_vars['model'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)

                        MSE = np.sqrt(((self.dict_vars['model'].values - self.dict_vars['buoy'].values)**2).mean()) 
                        RMSE = round(MSE,2)

                        bias = np.sqrt(((self.dict_vars['model'].values - np.mean(self.dict_vars['model'].values))**2).mean()) -\
                                np.sqrt(((self.dict_vars['buoy'].values - np.mean(self.dict_vars['buoy'].values))**2).mean())
                        bias = round(bias,2)

                        corr, _ = pearsonr(self.dict_vars['buoy'].values,self.dict_vars['model'].values)
                        corr=round(corr,2)

                        MBE = np.mean(self.dict_vars['model'].values - self.dict_vars['buoy'].values)
                        MBE = round(MBE,2)

                if 'hindcast_ERA5' in self.dict_vars.keys():
                                        
                        idx_nans=np.where(self.dict_vars['buoy'].isnull()==True)[0]
                        if len(idx_nans)>0:
                                dates_nans=self.dict_vars['buoy'].index[idx_nans]
                                self.dict_vars['hindcast_ERA5']=self.dict_vars['hindcast_ERA5'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['hindcast_CSFR']=self.dict_vars['hindcast_CSFR'].drop(index=dates_nans-pd.Timedelta(minutes=40))
                                self.dict_vars['buoy']=self.dict_vars['buoy'].drop(index=dates_nans)
                        
                        self.index_hindcast_era5=(self.dict_vars['hindcast_ERA5'][self.idate+relativedelta(hours=24):].index)
                        self.index_hindcast_csfr=(self.dict_vars['hindcast_CSFR'][self.idate+relativedelta(hours=24):].index)
                        self.index_buoy=self.dict_vars['buoy'].index-pd.Timedelta(minutes=40)
                        self.intersec_index_era5=self.index_hindcast_era5.intersection(self.index_buoy)
                        self.intersec_index_csfr=self.index_hindcast_csfr.intersection(self.index_buoy)
                        self.buoy_new_era5=self.dict_vars['buoy'][self.intersec_index_era5+pd.Timedelta(minutes=40)]
                        self.buoy_new_csfr=self.dict_vars['buoy'][self.intersec_index_csfr+pd.Timedelta(minutes=40)]
                        self.hindcast_new_era5=self.dict_vars['hindcast_ERA5'][self.intersec_index_era5]
                        self.hindcast_new_csfr=self.dict_vars['hindcast_CSFR'][self.intersec_index_csfr]

                        MSE_era5 = np.sqrt(((self.hindcast_new_era5.values - self.buoy_new_era5.values)**2).mean()) 
                        RMSE_era5 = round(MSE_era5,2)
                        bias_era5 = np.sqrt(((self.hindcast_new_era5.values - np.mean(self.hindcast_new_era5.values))**2).mean()) -\
                                np.sqrt(((self.buoy_new_era5.values - np.mean(self.buoy_new_era5.values))**2).mean())
                        bias_era5 = round(bias,2)
                        corr_era5, _ = pearsonr(self.buoy_new_era5.values,self.hindcast_new_era5.values)
                        corr_era5=round(corr_era5,2)

                        MBE_era5 = np.mean(self.hindcast_new_era5.values - self.buoy_new_era5.values)
                        MBE_era5 = round(MBE_era5,2)

                        MSE_csfr = np.sqrt(((self.hindcast_new_csfr.values - self.buoy_new_csfr.values)**2).mean()) 
                        RMSE_csfr = round(MSE_csfr,2)
                        bias_csfr = np.sqrt(((self.hindcast_new_csfr.values - np.mean(self.hindcast_new_csfr.values))**2).mean()) -\
                                np.sqrt(((self.buoy_new_csfr.values - np.mean(self.buoy_new_csfr.values))**2).mean())
                        bias_csfr = round(bias,2)
                        corr_csfr, _ = pearsonr(self.buoy_new_csfr.values,self.hindcast_new_csfr.values)
                        corr_csfr=round(corr_csfr,2)
                        MBE_csfr = np.mean(self.hindcast_new_csfr.values - self.buoy_new_csfr.values)
                        MBE_csfr = round(MBE_csfr,2)

                if id_buoy=='42058':
                        self.x_add=0.5
                        self.correction=0.1
                        self.y_add=1.4
                else:
                        self.x_add=0
                        self.correction=0
                        self.y_add=0
                for i in range (0,6):
                        self.ax=axes[i%2][i//2]

                        if label=='ctrl':
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,color=color,label=label,alpha=0.5)
                                self.ax.text(0.1, 2.8+self.y_add, 'Mean bias error=',fontsize=10)
                                self.ax.text(1.3+self.x_add-self.correction, 2.8+self.y_add, f'{MBE}',fontsize=10,color=color)
                                self.ax.text(0.1, 2.5+self.y_add, 'Correlation=',fontsize=10)
                                self.ax.text(0.9+self.x_add-self.correction, 2.5+self.y_add, f'{corr}',fontsize=10,color=color)
                                self.ax.text(0.1, 2.2+self.y_add, 'RMSE=',fontsize=10)
                                self.ax.text(0.6+self.x_add-self.correction, 2.2+self.y_add, f'{RMSE}',fontsize=10,color=color)

                        if label=='ctrl' and i==0:
                                self.ax.scatter(self.buoy_new_era5.values,self.hindcast_new_era5.values,
                                        s=35,color='pink',label='Hindcast ERA5',alpha=0.5)
                                self.ax.text(1.7+self.x_add, 2.8+self.y_add, f'{MBE_era5}',fontsize=10,color='pink')
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr_era5}',fontsize=10,color='pink')
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE_era5}',fontsize=10,color='pink')

                        elif label=='ctrl' and i==1:
                                self.ax.scatter(self.buoy_new_csfr.values,self.hindcast_new_csfr.values,
                                        s=35,color='olivedrab',label='Hindcast CSFR',alpha=0.5)
                                self.ax.text(1.85+self.x_add, 2.8+self.y_add, f'{MBE_csfr}',fontsize=10,color='olivedrab')
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr_csfr}',fontsize=10,color='olivedrab')
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE_csfr}',fontsize=10,color='olivedrab')

                        elif label=='cos2' and i==2:
                                label='$cos^2$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                self.ax.text(1.85+self.x_add, 2.8+self.y_add, f'{MBE}',fontsize=10,color=color)
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr}',fontsize=10,color=color)
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE}',fontsize=10,color=color)

                        elif label=='cos4' and i==3:
                                label='$cos^4$'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                self.ax.text(1.85+self.x_add, 2.8+self.y_add, f'{MBE}',fontsize=10,color=color)
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr}',fontsize=10,color=color)
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE}',fontsize=10,color=color)

                        elif label=='bim' and i==4:
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                self.ax.text(1.85+self.x_add, 2.8+self.y_add, f'{MBE}',fontsize=10,color=color)
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr}',fontsize=10,color=color)
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE}',fontsize=10,color=color)

                        elif label=='all' and i==5:
                                label='$cos^4$+bim'
                                self.ax.scatter(self.dict_vars['buoy'],self.dict_vars['model'],
                                                s=35,label=label,color=color,alpha=0.5)
                                self.ax.text(1.85+self.x_add, 2.8+self.y_add, f'{MBE}',fontsize=10,color=color)
                                self.ax.text(1.3+self.x_add, 2.5+self.y_add, f'{corr}',fontsize=10,color=color)
                                self.ax.text(1+self.x_add, 2.2+self.y_add, f'{RMSE}',fontsize=10,color=color)

                        self.ax.legend(loc='lower right')

                        if id_buoy=='42058':
                                self.ax.set(xlim=(0,4.5),ylim=(0,4.5),ylabel='Simulated $H_s$ [m]',xlabel='Observed $H_s$ [m]')
                        else:
                                self.ax.set(xlim=(0,3),ylim=(0,3),ylabel='Simulated $H_s$ [m]',xlabel='Observed $H_s$ [m]')

                        self.ax.plot(np.arange(-0.5,4.5,0.1),np.arange(-0.5,4.5,0.1),'--k')

                return axes


        def compare_another_conf(self,obj2,dict2,figs2,label,color):
                self.adding_variables=obj2.preparing_data()[0]
                self.dict_axes={buoy:[1] for buoy in self.buoys_id}
                for id_buoy in self.buoys_id:
                        self.dict_axes[id_buoy]=self.plotting_one_scatter(dict2[id_buoy],self.adding_variables,id_buoy,label,color)
                        
                        figs2[id_buoy].savefig(f'{self.plots_path}qqplot_{id_buoy}.png',dpi=1000,bbox_inches='tight',pad_inches=0.05)
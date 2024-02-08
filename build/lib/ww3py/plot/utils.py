import pandas as pd
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator


def ord_buoy_data(buoy_id):
        path='/home/fayalacruz/runs/reg_storm/info/plots/'+buoy_id+'h2020.txt'
        df_boya=pd.read_csv(path,header=0, delim_whitespace=True, dtype=object,
                na_values=['99.00',999,9999,99.,999.,9999.])[1:]
        fechas=df_boya[df_boya.columns[:5]].astype(float)
        fechas=fechas.rename(columns={"#YY": "year", "MM":"month","DD":'day','hh':'hour','mm':'minute'})
        fechas_with_nans=pd.to_datetime(fechas)
        hs_with_nans=df_boya['WVHT'].astype(np.float)
        dir_with_nans=df_boya['MWD'].astype(np.float)
        tp_with_nans=df_boya['DPD'].astype(np.float)
        wndir_with_nans=df_boya['WDIR'].astype(np.float)
        wspd_with_nans=df_boya['WSPD'].astype(np.float)
        hs=hs_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        dir=dir_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        tp=tp_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        wndir=wndir_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        wspd=wspd_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        fechas=fechas_with_nans[np.logical_not(np.isnan(hs_with_nans))]
        df_boya=pd.DataFrame(data={'hs':hs.values,'dir':dir.values,'tp':tp.values,'wndir':wndir.values,'wspd':wspd.values},index=fechas)
        return df_boya

def read_data_int_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=pd.DataFrame(data={'hs':ds_points.hs.values[:,ind],'dir':ds_points.th1p.values[:,ind],'tp':1/(ds_points.fp.values[:,ind])},index=time_points_ww3)
        return data_stations

def read_data_spec_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=ds_points.efth.values[:,ind,:,:]
        return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_stations

def read_data_src_stations(path,dim):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_snl_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                if dim =='2d':
                        data_snl_stations[station_id]=ds_points.sin.values[:,ind,:,:]
                else:
                        data_snl_stations[station_id]=np.sum(ds_points.sin.values[:,ind,:],axis=1)
        if dim =='2d':                
                return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_snl_stations
        else:
                return time_points_ww3,ds_points.frequency.values[:],data_snl_stations


def vert_colorbar(fig,ax,cf,pad,width,label):
        cax= fig.add_axes([ax.get_position().x1+pad,ax.get_position().y0,
                                        width,ax.get_position().height])
        cbar=plt.colorbar(cf,cax=cax,orientation="vertical")
        cbar.set_label(label)
        return cbar

def read_era5_buoys(path,lon_buoy,lat_buoy):
        era5_dtst=xr.load_dataset(path)
        lon=era5_dtst.longitude.values[:]
        lat=era5_dtst.latitude.values[:]
        time_points_era5=pd.to_datetime(era5_dtst.time.data)
        final_serie=[]
        for i in range(len(era5_dtst.time)):
                wind_speed=np.sqrt(((era5_dtst.u10.values[i,:,:])**2)+((era5_dtst.v10.values[i,:,:])**2))
                longitude,latitude=np.meshgrid(lon,lat)
                interp = RegularGridInterpolator((lon, lat), wind_speed.T)
                pts = np.array([lon_buoy,lat_buoy])
                final_serie.append(round(interp(pts).tolist()[0],3))
        a=pd.Series(final_serie,index=time_points_era5)
        return a
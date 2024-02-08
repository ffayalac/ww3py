import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from dateutil.relativedelta import relativedelta

def ord_buoy_data(buoy_id):
        path='/home/fayalacruz/runs/reg_storm_ctrl/info/plots/'+buoy_id+'h2020.txt'
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

def compute_azimuth(angle):
        if angle<=90:
                result = 90 - angle
        elif (angle>90) & (angle<=180):
                result = 270 + (180-angle)
        elif (angle>180) & (angle <=270):
                result = 180 + (270-angle)
        else:
                result = 90 + (360-angle)
        return result

def read_1d_spec_buoy(buoy_id):
        path=f'/home/fayalacruz/runs/reg_storm_ctrl/info/plots/{buoy_id}/{buoy_id}w2020.txt'
        df_boya=pd.read_csv(path, delim_whitespace=True, parse_dates={ 'date': ['#YY', 'MM', 'DD','hh','mm']})
        df_boya=df_boya.set_index('date')
        df_boya.index = pd.to_datetime(df_boya.index,format='%Y %m %d %H %M')

        freqs = np.array(list(map(lambda x: float(x),df_boya.columns)))
        times = df_boya.index
        specdens = df_boya.values

        return times,freqs,specdens


def read_2d_spec_buoy(buoy_id):
        url_sufixs = {'sw':'w','r1':'j','r2':'k','a1':'d','a2':'i'}
        dfs={}
        for label,suf in url_sufixs.items():
                path=f'/home/fayalacruz/runs/reg_storm_ctrl/info/plots/{buoy_id}/{buoy_id}{suf}2020.txt'
                df_boya=pd.read_csv(path, delim_whitespace=True, parse_dates={ 'date': ['#YY', 'MM', 'DD','hh','mm']})
                df_boya=df_boya.set_index('date')
                df_boya.index = pd.to_datetime(df_boya.index,format='%Y %m %d %H %M')
                dfs[label]=df_boya
                if suf == 'w':
                        freqs = np.array(list(map(lambda x: float(x),df_boya.columns)))

        times = dfs['sw'].index
        spshape = (len(times), len(freqs), 1)
        specdens = dfs['sw'].values.reshape(spshape)
        swdir1=dfs['a1'].values.reshape(spshape)
        swdir2=dfs['a2'].values.reshape(spshape)
        swr1=dfs['r1'].values.reshape(spshape)
        swr2=dfs['r2'].values.reshape(spshape)


        dirs=np.arange(0, 360, 10)

        dirmat = dirs.reshape((1, 1, -1))
        IPI = 1/np.pi
        D2R = np.pi/180
        D_fd = (
                IPI * (0.5 + swr1 * np.cos(D2R * (dirmat - swdir1)) + swr2 * np.cos(2 * D2R * (dirmat - swdir2)))* D2R
                )
        S = specdens * D_fd
        return times,freqs,dirs,S


def read_data_int_stations(path):
        ds_points=xr.load_dataset(path)
        data_stations={}
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=pd.DataFrame(data={'hs':ds_points.hs.values[:,ind],'dirp':ds_points.th1p.values[:,ind],'tp':1/(ds_points.fp.values[:,ind]),
                                                             'fp':ds_points.fp.values[:,ind]},index=time_points_ww3)
        return data_stations

def read_data_extra_stations(path):
        ds_points=xr.load_dataset(path)
        data_stations={}
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=pd.DataFrame(data={'dpt':ds_points.dpt.values[:,ind],'cur':ds_points.cur.values[:,ind],'wnd':ds_points.wnd.values[:,ind],
                                                             'wnddir':ds_points.wnddir.values[:,ind]},index=time_points_ww3)
        return data_stations

def read_data_spec_1d_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=ds_points.f.values[:,ind,:]
        return time_points_ww3,ds_points.frequency.values[:],data_stations


def read_data_spec_stations(path):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                data_stations[station_id]=ds_points.efth.values[:,ind,:,:]
        return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_stations

def read_data_src_stations(path,dim,term):
        ds_points=xr.load_dataset(path)
        time_points_ww3=pd.to_datetime(ds_points.time.data)
        data_src_stations={}
        for ind,station in enumerate(ds_points.station_name.data):
                pre_station_id=b''.join(station)
                station_id=pre_station_id.decode('utf-8')
                if dim =='2d':
                        data_src_stations[station_id]=ds_points[term].values[:,ind,:,:]
                else:
                        data_src_stations[station_id]=ds_points[term].values[:,ind,:]
        if dim =='2d':                
                return time_points_ww3,ds_points.frequency.values[:],np.radians(ds_points.direction.values[:]),data_src_stations
        elif dim=='1d':
                return time_points_ww3,ds_points.frequency.values[:],data_src_stations

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

def read_spatial_data(path):
        ds_spatial=xr.load_dataset(path)
        lon_spatial=ds_spatial.longitude.values
        lat_spatial=ds_spatial.latitude.values
        time_points_ww3=pd.to_datetime(ds_spatial.time.data)
        return time_points_ww3,lon_spatial,lat_spatial,ds_spatial.utaw.values,ds_spatial.vtaw.values


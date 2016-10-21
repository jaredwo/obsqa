from .util import grt_circle_dist
from scipy import stats
from scipy.stats.stats import pearsonr
import itertools
import numpy as np
import pandas as pd
import xray as xr

def qa_dly_tmin_tmax_naught(obs):
    '''Flag observations with erroneous zeros US stations: Tmax and Tmin = -17.8C (0F)
    Non-US: Tmax and Tmin = 0 C
    
    Requires Tmin and Tmax
    '''
    mask_us = ((obs.tmin.round(1) == -17.8) & (obs.tmax.round(1) == -17.8))
    mask_nonus = ((obs.tmin == 0) & (obs.tmax == 0))
    mask_final = ((mask_us) | (mask_nonus))
    
    i_flg = obs.index[mask_final]
    
    return {'tmin':i_flg, 'tmax':i_flg}

def qa_dly_elem_dup_year(obs, elem):
    '''Flag duplicate observations between years
    '''
    
    obs_elem = obs[[elem]].copy()
    
    # Add year day column
    obs_elem['yday'] = obs_elem.index.dayofyear
    
    # Mark observations that have the same obs value and year day
    dup_mask = obs_elem.duplicated(subset=[elem, 'yday'], keep=False)
    
    # Get dup observations and count the number per year
    obs_dup = obs_elem[dup_mask]
    obs_dup = obs_dup[[elem, 'yday']].resample("A", how='count')
    
    # Limit candidate dup year to those that have actual observations
    # and have at least 365 dups
    time_dup = obs_dup.index[((obs_dup[elem] > 0) & (obs_dup.yday >= 365))]    
    yrs = time_dup.year.astype(np.str)
    
    # Loop through all unique pairs of year and note which years have all
    # duplicate values
    dup_yrs = []
    
    for yr1, yr2 in itertools.combinations(yrs, 2):
        
        # Get observations for year 1 and year 2
        elem_yr1 = obs_elem.loc[yr1]
        elem_yr2 = obs_elem.loc[yr2]
        
        dup_mask = pd.concat((elem_yr1, elem_yr2)).duplicated(keep=False)
                                    
        if dup_mask.all():
                            
            dup_yrs.extend((yr1, yr2))
                
    # Get all duplicate years            
    dup_yrs = np.unique(np.array(dup_yrs).astype(np.int))
     
    # Get all dates for duplicate years that should be flagged
    dates = pd.Series(obs_elem.index, index=obs_elem.index.year).loc[dup_yrs]
    
    # Remove dates that have nan values
    dates = obs.loc[dates, elem].dropna().index

    return dates

def qa_dly_elem_dup_year_month(obs, elem):
    '''Flag months in a single year that have duplicate values.
    '''
    
    obs_elem = obs[[elem]].copy()
    
    # Add year and month day columns
    obs_elem['year'] = obs_elem.index.year
    obs_elem['day'] = obs_elem.index.day
    
    # Mark observations that have the same obs value, year and month day
    dup_mask = obs_elem.duplicated(subset=[elem, 'year', 'day'], keep=False)
    
    # Get dup observations and count the number per month
    obs_dup = obs_elem[dup_mask]
    obs_dup = obs_dup[[elem, 'year']].resample("M", how='count')
    
    # Limit candidate dup months to those that have actual observations
    # and have at least 28 dups (minimum # days in a month)
    time_dup = obs_dup.index[((obs_dup[elem] > 0) & (obs_dup.year >= 28))]
    time_dup = pd.DataFrame({'month':time_dup.month},
                            index=time_dup.year)
    
    # Confirm candidate dup months 
    dup_yr_mths = []
    
    for yr in np.unique(time_dup.index.values):
        
        try:
            mths_yr = time_dup.month.loc[yr].values
        except AttributeError as e:
            if e.args[0] == "'numpy.int32' object has no attribute 'values'":
                # only one entry for this year. Continue to next.
                continue
            else:
                raise
        
        for mth1,mth2 in itertools.combinations(mths_yr,2):
            
            yrmth1 = "%d-%.2d"%(yr, mth1)
            yrmth2 = "%d-%.2d"%(yr, mth2)
            elem_mth1 = obs_elem.loc[yrmth1, elem]
            elem_mth2 = obs_elem.loc[yrmth2, elem]
            
            # Compare up to last day of shortest month
            last_day = np.min([elem_mth1.size, elem_mth2.size])
            elem_mth1 = elem_mth1.iloc[0:last_day]
            elem_mth2 = elem_mth2.iloc[0:last_day]
                        
            # Months are duplicates if their values are all the same
            # Use duplicated function instead of direct comparison so
            # that matching NAs are considered dups
            dup_mask = pd.concat((elem_mth1, elem_mth2)).duplicated(keep=False)
                                    
            if dup_mask.all():
                                
                dup_yr_mths.extend((yrmth1, yrmth2))
        
    dup_yr_mths = np.unique(dup_yr_mths)
    
    try:
        dates = np.concatenate([obs_elem.loc[yrmth].index.values
                                for yrmth in dup_yr_mths])
    except ValueError as e:
        
        if e.args[0] == 'need at least one array to concatenate':
            dates = []
        else:
            raise
        
    # Remove dates that have nan values
    dates = obs.loc[dates, elem].dropna().index
        
    return dates
  
def qa_dly_elem_dup_month(obs, elem):
    '''
    Flag duplicate observations for same calendar month in different years
    '''
    
    obs_elem = obs[[elem]].copy()
    
    # Add year and month day columns
    obs_elem['month'] = obs_elem.index.month
    obs_elem['day'] = obs_elem.index.day
    
    # Mark observations that have the same obs value, month and month day
    dup_mask = obs_elem.duplicated(subset=[elem, 'month', 'day'], keep=False)
    
    # Get dup observations and count the number per month
    obs_dup = obs_elem[dup_mask]
    obs_dup = obs_dup[[elem, 'month']].resample("M", how='count')
    
    # Limit candidate dup months to those that have actual observations
    # and have at least 28 dups (minimum # days in a month)
    time_dup = obs_dup.index[((obs_dup[elem] > 0) & (obs_dup.month >= 28))]
    time_dup = pd.DataFrame({'year':time_dup.year},
                            index=time_dup.month)
    
    # Confirm candidate dup months
    dup_mths = []
    
    for mth in np.unique(time_dup.index.values):
        
        try:
            yrs_mth = time_dup.year.loc[mth].values
        except AttributeError as e:
            if e.args[0] == "'numpy.int32' object has no attribute 'values'":
                # only one entry for this month. Continue to next.
                continue
            else:
                raise
        
        for yr1,yr2 in itertools.combinations(yrs_mth,2):
            
            mthyr1 = "%d-%.2d"%(yr1, mth)
            mthyr2 = "%d-%.2d"%(yr2, mth)
            elem_mthyr1 = obs_elem.loc[mthyr1]
            elem_mthyr2 = obs_elem.loc[mthyr2]
            
            # Compare up to last day of shortest month
            last_day = np.min([elem_mthyr1.size, elem_mthyr2.size])
            elem_mthyr1 = elem_mthyr1.iloc[0:last_day]
            elem_mthyr2 = elem_mthyr2.iloc[0:last_day]
            
            # Months are duplicates if their values are all the same
            # Use duplicated function instead of direct comparison so
            # that matching NAs are considered dups
            dup_mask = pd.concat((elem_mthyr1,
                                  elem_mthyr2)).duplicated(keep=False)
            
            if dup_mask.all():
                                
                dup_mths.extend((mthyr1, mthyr2))
        
    dup_mths = np.unique(dup_mths)
    
    try:
        dates = np.concatenate([obs_elem.loc[yrmth].index.values
                                for yrmth in dup_mths])
    except ValueError as e:
        
        if e.args[0] == 'need at least one array to concatenate':
            dates = []
        else:
            raise
        
    # Remove dates that have nan values
    dates = obs.loc[dates, elem].dropna().index
            
    return dates

def qa_dly_tmin_tmax_dup_within_month(obs):
    '''Flag all days in months with 10 or more days that Tmax=Tmin
    '''
    
    mask_dup = obs.tmin == obs.tmax
    dup_cnt = mask_dup.resample("M", how='sum')
    dup_cnt = dup_cnt[dup_cnt >= 10]
    
    mths_dup = dup_cnt.index.strftime('%Y-%m')
    
    try:
        obs_dup = pd.concat([obs.loc[a_mth] for a_mth in mths_dup])
        dates_tmin = obs_dup.tmin.dropna().index
        dates_tmax = obs_dup.tmax.dropna().index
    except ValueError as e:
        
        if e.args[0] == 'No objects to concatenate':
            # No flagged dates. Send back empty DatetimeIndex
            dates_tmin = obs.loc[[]].index
            dates_tmax = dates_tmin
        else:
            raise
        
    return {'tmin':dates_tmin, 'tmax':dates_tmax}

def qa_dly_elem_streak(obs, elem):
    '''
    Flag 20 or more consecutive observations
    '''
    
    # Remove missing values.
    # They are skipped when identifying a streak
    obs_elem = obs[elem].dropna()
        
    # Get unique values. Only consider those that are duplicated
    uvals = obs_elem[obs_elem.duplicated(keep=False)].unique()
    
    streak_dates = []
    
    for aval in uvals:
        
        is_equal = (obs_elem == aval).values.astype(np.int)
        bounded = np.hstack(([0],is_equal,[0]))
        difs = np.diff(bounded)
        run_start, = np.where(difs > 0)
        run_end, = np.where(difs < 0)
        run_size = run_end-run_start
        i_streaks, = np.where(run_size >= 20)
        run_start = np.take(run_start, i_streaks)
        run_end = np.take(run_end, i_streaks)
        
        for a_start,a_end in zip(run_start, run_end):
            streak_dates.append(obs_elem.iloc[a_start:a_end].index.values)
    
    try:
    
        streak_dates = np.concatenate(streak_dates)
    
    except ValueError as e:
        
        if e.args[0] != 'need at least one array to concatenate':
            raise
        
    return obs_elem.loc[streak_dates].index

def qa_dly_elem_imposs_value(obs, elem, val_min, val_max):
    '''
    Flag observations that are outside the bounds of world records
    Tair: -89.4, 57.7
    '''

    mask_imposs = np.logical_or(obs[elem] < val_min, obs[elem] > val_max)
    
    return obs.index[mask_imposs]

def qa_dly_elem_gap(obs, elem, gap_thres):
    '''
    Examine frequency distributions of elem for calendar months and flag observations in
    a distribution's tails that are unrealistically separated from the rest of the observations
    gap_thres tair: 10 deg C
    '''
    
    obs_elem = obs[elem].dropna()
    uniq_mths = np.unique(obs_elem.index.month)
    gap_dates = []
    
    def _get_gap_bounds(vals):
        
        vals_sorted = np.sort(vals)
    
        val_median = np.median(vals_sorted)
        val_top = vals_sorted[vals_sorted >= val_median]
        val_bottom = vals_sorted[vals_sorted <= val_median][::-1]
    
        gap_mask_top = np.ediff1d(val_top, to_begin=[0]) >= gap_thres
    
        if np.any(gap_mask_top):
            bnds_top = val_top[gap_mask_top][0]
        else:
            bnds_top = None
    
        gap_mask_bottom = np.abs(np.ediff1d(val_bottom, to_begin=[0])) >= gap_thres
    
        if np.any(gap_mask_bottom):
            bnds_bottom = val_bottom[gap_mask_bottom][0]
        else:
            bnds_bottom = None
    
        return (bnds_bottom, bnds_top)

    for mth in uniq_mths:

        elem_mth = obs_elem[obs_elem.index.month==mth]

        elem_min, elem_max = _get_gap_bounds(elem_mth.values)

        if elem_min is not None:
            
            gap_dates.append(elem_mth.index[elem_mth <= elem_min].values)
        
        if elem_max is not None:
        
            gap_dates.append(elem_mth.index[elem_mth >= elem_max].values)

    try:
    
        gap_dates = np.sort(np.concatenate(gap_dates))
    
    except ValueError as e:
        
        if e.args[0] != 'need at least one array to concatenate':
            raise
        
    return obs.loc[gap_dates].index

def qa_dly_elem_clim_outlier(obs, elem, std_thres, min_obs):
    '''
    Check for Tmin outliers based on z-score value > std_thres standard deviations of 15-day climate norm
    Must have more than min_obs values within 15-day period for this check to run.
    Tair: std_thres = 6.0 | min_obs = 100
    
    '''
    
    idx_name = obs.index.name if obs.index.name is not None else 'index'
    
    # Drop missing
    obs_elem = pd.DataFrame(obs[elem].dropna())
    
    if len(obs_elem) < min_obs:
        
        # Not enough observations to run outlier check for any month
        # Return empty index
        return obs.loc[[]].index
    
    # Add month day variable    
    obs_elem['mthday'] = obs_elem.index.strftime('%m-%d')
    # Set any leap month days to March 1st
    obs_elem.loc[obs_elem.mthday=='02-29','mthday'] = '03-01' 
    # Set index to month day
    obs_elem = obs_elem.reset_index(drop=False).set_index('mthday')
    
    dates, rngs_mthday = _DateRngs.date_rngs()
            
    # Build 15-day norms for every month day in dates
    norms = np.empty([len(rngs_mthday), 2])
    norms.fill(np.nan)
        
    for i, a_rng in enumerate(rngs_mthday):
        
        try:
            elem_rng = obs_elem.loc[a_rng, elem].values
        except KeyError:
            # No observations for year days in a_rng
            # Continue to next
            continue
        
        if elem_rng.size >= min_obs:
            
            norms[i,:] = _biweight_mean_std(elem_rng)
            
    norms = pd.DataFrame(norms, index=dates.strftime('%m-%d'),
                         columns=['obs_mean','obs_std'] )
    
    # Add norms to tmin dataframe
    obs_elem = obs_elem.join(norms, how='left')
    
    # Calculate zscores for each date
    obs_elem['zscore'] = ((obs_elem[elem]-obs_elem.obs_mean) / obs_elem.obs_std).abs()
    
    # Any observations with zscore >= std_thres is considered an outlier
    outlier_dates = obs_elem.loc[obs_elem.zscore >= std_thres, idx_name].values
    
    return obs.loc[outlier_dates].index

def _biweight_mean_std(X):
    '''
    Calculates more robust mean/std for climate data
    Used by Durre et al. 2010 referencing Lanzante 1996 Appendix B
    '''

    # mean
    c = 7.5
    M = np.median(X)
    MAD = np.median(np.abs(X - M))

    if MAD == 0:
        # return normal mean, std. biweight cannot be calculated
        return np.mean(X), np.std(X, ddof=1)

    u = (X - M) / (c * MAD)
    u[np.abs(u) >= 1.0] = 1.0
    Xbi = M + (np.sum((X - M) * (1.0 - u ** 2) ** 2) /
               np.sum((1 - u ** 2) ** 2))

    # std
    n = X.size
    Sbi = (((n * np.sum(((X - M) ** 2) * (1 - u ** 2) ** 4)) ** 0.5) / 
           np.abs(np.sum((1 - u ** 2) * (1 - (5 * u ** 2)))))

    return Xbi, Sbi

class _DateRngs(object):
    
    _date_rngs = None
    
    @classmethod
    def date_rngs(cls):
        
        if cls._date_rngs is None:
            
            #Get all dates for a standard non-leap year
            dates = pd.date_range('2013-01-01', '2013-12-31')
                
            srt_dates = dates - pd.Timedelta(days=7)
            end_dates = dates + pd.Timedelta(days=7)
            rngs_mthday = [pd.date_range(a_start,a_end).strftime('%m-%d')
                           for a_start,a_end in zip(srt_dates, end_dates)]
            
            cls._date_rngs = (dates, rngs_mthday)
        
        return cls._date_rngs
        
def qa_dly_tmin_tmax_internal_inconsist(obs):
    '''
    Flag inconsistent Tmin observations. Requires Tmin and Tmax.
    Unlike the set of inconsistent checks described by Durre et al. (2010), only check for
    Tmax < Tmin on same day and do not use a 1 deg buffer. The 1 deg buffer and
    other inconsistent checks were found to have too many false positives in SNOTEL
    and RAWS data. Also do not include checks that use time of observation.
    '''
    dates_flg = obs.index[obs.tmin > obs.tmax]
    
    return {'tmin': dates_flg, 'tmax': dates_flg}

def qa_dly_elem_spike_dip(obs, elem, swing_thres):
    '''Check for unrealistic swings in temperature on adjacent days
    Tair swing_thres: 25.0
    '''
    
    elem_prev = obs[elem].tshift(1)
    elem_prev.name = 'elem_prev'
    elem_next = obs[elem].tshift(-1)
    elem_next.name = 'elem_next'
    obs_elem = pd.DataFrame(obs[elem].copy())
    obs_elem = obs_elem.join(elem_prev, how='left').join(elem_next, how='left')
    
    spikedip_dates = obs_elem.index[((obs_elem[elem]-obs_elem.elem_next).abs() >= swing_thres) &
                                    ((obs_elem[elem]-obs_elem.elem_prev).abs() >= swing_thres)].values
    
    return obs.loc[spikedip_dates].index
    
def qa_dly_tmin_tmax_lagrange_inconsist(obs):
    '''
    Check for differences in excess of 40C between Tmin and coolest Tmax in current/adjacent days.
    Requires Tmax
    '''
    
    # Perform Tmin check using prev,cur,next Tmax for each day
    tmax_prev = obs.tmax.tshift(1)
    tmax_prev.name = 'tmax_prev'
    tmax_next = obs.tmax.tshift(-1)
    tmax_next.name = 'tmax_next'
    tmax = pd.DataFrame(obs.tmax.copy())
    tmax = tmax.join(tmax_prev, how='left').join(tmax_next, how='left')
    tmax['tmax_min'] = tmax.min(axis=1, skipna=False)
    
    tmin = pd.DataFrame(obs.tmin.copy())
    tmin = tmin.join(tmax['tmax_min'],how='left')
        
    inconsist_dates_tmin = tmin.index[(tmin.tmin - tmin.tmax_min).abs() > 40]
    # Also flag the 3 Tmax values surrounding the flagged tmin values
    inconsist_dates_tmax = np.concatenate((inconsist_dates_tmin.values,
                                           (inconsist_dates_tmin - pd.Timedelta(days=1)).values,
                                           (inconsist_dates_tmin + pd.Timedelta(days=1)).values))
    
    # Perform Tmax check using prev,cur,next Tmin for each day
    tmin_prev = obs.tmin.tshift(1)
    tmin_prev.name = 'tmin_prev'
    tmin_next = obs.tmin.tshift(-1)
    tmin_next.name = 'tmin_next'
    tmin = pd.DataFrame(obs.tmin.copy())
    tmin = tmin.join(tmin_prev, how='left').join(tmin_next, how='left')
    tmin['tmin_max'] = tmin.max(axis=1, skipna=False)
    
    tmax = pd.DataFrame(obs.tmax.copy())
    tmax = tmax.join(tmin['tmin_max'],how='left')
        
    inconsist_dates_tmax2 = tmax.index[(tmax.tmax - tmax.tmin_max).abs() > 40]
    # Also flag the 3 Tmin values surrounding the flagged tmin values
    inconsist_dates_tmin2 = np.concatenate((inconsist_dates_tmax2.values,
                                           (inconsist_dates_tmax2 - pd.Timedelta(days=1)).values,
                                           (inconsist_dates_tmax2 + pd.Timedelta(days=1)).values))
    
    inconsist_dates_tmin = np.unique(np.concatenate((inconsist_dates_tmin,
                                                     inconsist_dates_tmin2)))
    inconsist_dates_tmax = np.unique(np.concatenate((inconsist_dates_tmax,
                                                     inconsist_dates_tmax2)))
    
    inconsist_dates_tmin = obs.loc[obs.index.isin(inconsist_dates_tmin)].index
    inconsist_dates_tmax = obs.loc[obs.index.isin(inconsist_dates_tmax)].index
    
    return {'tmin':inconsist_dates_tmin, 'tmax':inconsist_dates_tmax}

def qa_dly_tmin_spatial_regress(stn_id, stn_lon, stn_lat, obs, xr_ds, rm_stnids=None):
    '''
    Check for Tmin observations that are significantly different
    than surrounding neighbor stations (i.e.--not spatially consistent)
    via a spatial regression approach.
    '''
    
    if rm_stnids is None:
        rm_stnids = [stn_id]
    else:
        rm_stnids = rm_stnids + [stn_id]
    
    ngh_longitude = xr_ds.longitude
    ngh_latitude = xr_ds.latitude
    ngh_stnid = xr_ds.station_id
            
    mask_ngh = ~ngh_stnid.isin(rm_stnids)
    ngh_longitude = ngh_longitude[mask_ngh]
    ngh_latitude = ngh_latitude[mask_ngh]
    ngh_stnid = ngh_stnid[mask_ngh]
    
    d = grt_circle_dist(stn_lon, stn_lat, ngh_longitude.values,
                        ngh_latitude.values)

    mask_ngh = d <= 75.0
    
    ngh_stnid = ngh_stnid[mask_ngh]
    
    dates_flged = []

    if ngh_stnid.size >= 3:

        start_date = obs.index.min().to_datetime()
        end_date = obs.index.max().to_datetime()
        yrs = np.unique(obs.index.year)
        mths = np.unique(obs.index.month)
        
        ngh_obs = xr_ds.tmin.loc[start_date:end_date, list(ngh_stnid)]
        atmin = obs.tmin.reindex(ngh_obs.time.values)
        atmin = np.reshape(atmin, (atmin.size,1))
        atmin = xr.DataArray(atmin, coords=[obs.index.values,[stn_id]],
                             dims=['time','station_id'],name='tmin')
        
        atmin = xr.concat((atmin, ngh_obs), dim='station_id')
        
        for yr,mth in itertools.product(yrs,mths):
                                            
            start_date_mth = pd.Timestamp('%d-%.2d-01'%(yr,mth)) 
            end_date_mth = (pd.Timestamp('%d-%.2d-01'%(yr,mth)) +
                            pd.DateOffset(months=1) - pd.DateOffset(days=1))             
            
            start_date_mthwin =  start_date_mth - pd.DateOffset(days=15)
            end_date_mthwin =  end_date_mth + pd.DateOffset(days=15)
            
            atmin_mth = atmin.loc[start_date_mthwin:end_date_mthwin,:]
            
            # Only proceed with check if target station has at least 40
            # observations in the month window 
            if int(atmin_mth[:,0].count()) >= 40:
                
                # Subset to stations that have observations that overlap with
                # target station observations on at least 40 days
                n_overlap = ((~atmin_mth[:,0].isnull()) &
                             (~atmin_mth.isnull())).sum(axis=0)
                
                atmin_mth = atmin_mth[:, n_overlap >= 40]
                
                # Only proceed with check if there are at least 3 potential
                # neighbors
                nnghs = atmin_mth.shape[1] - 1
                
                if nnghs >= 3:
                    
                    # Calculate index of agreement and linear model
                    # for each neighbor
                    mask_valid = ~atmin_mth.isnull()
                    
                    ioa_mod = pd.DataFrame(np.empty((nnghs+1,3)),
                                           columns=['ioa','intercept','slope'],
                                           index=atmin_mth.station_id)
                    # Put dummy 2.0 ioa, and intercept/slope for station itself
                    ioa_mod.iloc[0] = 2.0,0,1.0
                    
                    for ngh_id in atmin_mth.station_id[1:].values:
                        
                        mask_ngh = (mask_valid[:,0]) & (mask_valid.loc[:, ngh_id])
                        
                        tmin_mth_stn = atmin_mth[mask_ngh,0]
                        tmin_mth_ngh = atmin_mth.loc[mask_ngh,ngh_id]
                    
                        ioa_mod.loc[ngh_id,'ioa'] = _calc_ioa_d1(tmin_mth_stn.values,
                                                                 tmin_mth_ngh.values)
                        
                        slope, intercept = stats.linregress(tmin_mth_ngh.values,
                                                            tmin_mth_stn.values)[0:2]
                        
                        ioa_mod.loc[ngh_id,'slope'] = slope 
                        ioa_mod.loc[ngh_id,'intercept'] = intercept                                                              
                    
                    # Sort by index of agreement
                    idx_ioa = np.argsort(ioa_mod.ioa.values)[::-1]
                    ioa_mod = ioa_mod.iloc[idx_ioa]
                    atmin_mth = atmin_mth[:,idx_ioa]
                    mask_valid = mask_valid[:,idx_ioa]
                    
                    # Get rid of days with no observation at target station
                    atmin_mth = atmin_mth[mask_valid[:,0].values,:]
                    mask_valid = mask_valid[mask_valid[:,0].values,:]
                    
                    tmin_pred = atmin_mth.loc[start_date_mth:end_date_mth,
                                              stn_id].copy()
                    tmin_pred[:] = np.nan
                    for a_date in tmin_pred.time.values:
                                                    
                        tmin_stn_day = atmin_mth.loc[a_date,stn_id]
                        tmin_ngh_day = atmin_mth.loc[(a_date-pd.Timedelta(days=1)):
                                                     (a_date+pd.Timedelta(days=1)),
                                                     atmin_mth.station_id[1:]]
                        
                        difs = np.abs(tmin_stn_day-tmin_ngh_day)
                        
                        # Drop stations that have no obs for the date
                        # and only proceed if there are at least 3 nghs
                        mask_ngh_obs = difs.count(dim='time') != 0
                        if mask_ngh_obs.sum() >= 3:
                        
                            difs = difs[:,mask_ngh_obs]
                            tmin_ngh_day = tmin_ngh_day[:,mask_ngh_obs]
                            
                            # Select obs for each station in 3-day window that
                            # is closest to obs for target station for the day
                            tmin_ngh_day = tmin_ngh_day.isel_points(time=difs.argmin(dim='time', skipna=True).values,
                                                                    station_id=np.arange(tmin_ngh_day.shape[1]))[0:7]
                            
                            tmin_pred_day = (ioa_mod.loc[tmin_ngh_day.station_id,'intercept'] +
                                             (tmin_ngh_day.values*ioa_mod.loc[tmin_ngh_day.station_id,'slope']))
                            wgts = ioa_mod.loc[tmin_ngh_day.station_id,'ioa']
                            wgts = wgts/wgts.sum()
                            
                            tmin_pred.loc[a_date] = np.average(tmin_pred_day.values,
                                                               weights=wgts.values)
                    
                    tmin_obs = atmin_mth.loc[start_date_mth:end_date_mth, stn_id]
                    
                    mask_pred = ~tmin_pred.isnull()
                    
                    tmin_pred = tmin_pred[mask_pred]
                    tmin_obs = tmin_obs[mask_pred]
                    r = pearsonr(tmin_obs.values,tmin_pred.values)[0]
                    
                    if r >= 0.8:
                        
                        resid = tmin_obs - tmin_pred
                        std_resid =  (resid - resid.mean())/resid.std(ddof=1)
                        
                        mask_flg = ((np.abs(resid) >= 8) &
                                    (np.abs(std_resid) >= 4.0))
                        
                        if mask_flg.any():
                            
                            dates_flged.append(tmin_obs.time[mask_flg].values)
                                       
    try:
    
        dates_flged = np.concatenate(dates_flged)
    
    except ValueError as e:
        
        if e.args[0] != 'need at least one array to concatenate':
            raise
                            
                            
    return obs.loc[dates_flged].index    
                
def _calc_ioa_d1(o,p):
    '''
    Calculate the index of agreement d1 between observed and predicted values.
    d1 ranges from 0.0 - 1.0 with values closer to 1.0 indicating better
    performance. 
    
    References:
    
    Willmott, C. J., S. G. Ackleson, R. E. Davis, J. J. Feddema, K. M. Klink,
    D. R. Legates, J. ODonnell, and C. M. Rowe (1985), Statistics for the 
    evaluation and comparison of models, J. Geophys. Res., 90(C5), 8995-9005,
    doi:10.1029/JC090iC05p08995.

    Legates, D. R., and G. McCabe (1999), Evaluating the use of goodness-of-fit 
    Measures in hydrologic and hydroclimatic model validation, Water Resour. Res.,
    35(1), PP. 233-241, doi:199910.1029/1998WR900018.

    Willmott, C. J., S. M. Robeson, and K. Matsuura (2012), A refined index of model
    performance, Int. J. Climatol., 32(13), 2088-2094, doi:10.1002/joc.2419.
    
    
    Parameters
    ----------
    o : ndarray
        Array of observations
    p : ndarray
        Array of predictions
        
    Returns
    -------
    d1 : float
        The d1 index of agreement
    '''
    
    o_mean = np.mean(o)
    denom = np.sum(np.abs(p - o_mean) + np.abs(o - o_mean))
        
    d1 = 1.0 - (np.sum(np.abs(p - o)) / denom)
    
    return d1

def qa_dly_tmin_tmax_mega_inconsist(obs):
    '''
    Last check that looks for Tmin values higher than highest Tmax for a calendar month
    and Tmax values lower than lowest Tmin value for a calendar month
    '''
    
    obs1 = obs.copy()
    obs1['month'] = obs1.index.month
    
    mth_stats = obs1.groupby('month').agg({'tmax':['count','max'],
                                           'tmin':['count','min']})
    mth_stats.columns = ['_'.join(col) for col in mth_stats.columns.values]
    mth_stats = mth_stats.reset_index()
    
    obs1 = obs1.merge(mth_stats, how='left', on='month')
    
    mask_flg_tmin = ((obs1.tmin > obs1.tmax_max) &
                     (obs1.tmax_count >= 140))
    
    mask_flg_tmax = ((obs1.tmax < obs1.tmin_min) &
                     (obs1.tmin_count >= 140))  
    
    return {'tmin':obs.index[mask_flg_tmin.values],
            'tmax':obs.index[mask_flg_tmax.values]}

def run_qa_dly_tmin_tmax_durre(obs):
    '''
    Run Tmin/Tmax quality assurance checks that do not require
    neighboring station data in the order specified by:
    
        Durre, I., M. J. Menne, B. E. Gleason, T. G. Houston, and R. S. Vose. 2010. 
        Comprehensive Automated Quality Assurance of Daily Surface Observations. 
        Journal of Applied Meteorology and Climatology 49:1615-1633.
        
    Parameters
    ----------
    
    Returns
    -------
    '''
    
    obs1 = obs.copy()
    obs1['flag_tmin'] = np.nan
    obs1['flag_tmax'] = np.nan
    
    def update_flgs(dates_flg, flg, elem):
        
        if len(dates_flg) > 0:
            
            obs1.loc[dates_flg, 'flag_%s'%elem] = flg
            obs1.loc[dates_flg, elem] = np.nan
    
    def update_flgs_tmin_tmax(dates_flg, flg):
        
        update_flgs(dates_flg['tmin'], flg, 'tmin')
        update_flgs(dates_flg['tmax'], flg, 'tmax')
    
    
    update_flgs_tmin_tmax(qa_dly_tmin_tmax_naught(obs1), 'N')
    
    update_flgs(qa_dly_elem_dup_year(obs1, 'tmin'), 'D', 'tmin')
    update_flgs(qa_dly_elem_dup_year(obs1, 'tmax'), 'D', 'tmax')
    
    update_flgs(qa_dly_elem_dup_year_month(obs1, 'tmin'), 'D', 'tmin')
    update_flgs(qa_dly_elem_dup_year_month(obs1, 'tmax'), 'D', 'tmax')
    
    update_flgs(qa_dly_elem_dup_month(obs1, 'tmin'), 'D', 'tmin')
    update_flgs(qa_dly_elem_dup_month(obs1, 'tmax'), 'D', 'tmax')

    update_flgs_tmin_tmax(qa_dly_tmin_tmax_dup_within_month(obs1), 'D')

    update_flgs(qa_dly_elem_imposs_value(obs1, 'tmin', -89.4, 57.7), 'X', 'tmin')
    update_flgs(qa_dly_elem_imposs_value(obs1, 'tmax', -89.4, 57.7), 'X', 'tmax')

    update_flgs(qa_dly_elem_streak(obs1, 'tmin'), 'K', 'tmin')
    update_flgs(qa_dly_elem_streak(obs1, 'tmax'), 'K', 'tmax')

    update_flgs(qa_dly_elem_gap(obs1, 'tmin', 10.0), 'G', 'tmin')
    update_flgs(qa_dly_elem_gap(obs1, 'tmax', 10.0), 'G', 'tmax')

    update_flgs(qa_dly_elem_clim_outlier(obs1, 'tmin', 6.0, 100), 'O', 'tmin')
    update_flgs(qa_dly_elem_clim_outlier(obs1, 'tmax', 6.0, 100), 'O', 'tmax')

    update_flgs_tmin_tmax(qa_dly_tmin_tmax_internal_inconsist(obs1), 'I')
    
    update_flgs(qa_dly_elem_spike_dip(obs1, 'tmin', 25.0), 'I', 'tmin')
    update_flgs(qa_dly_elem_spike_dip(obs1, 'tmax', 25.0), 'I', 'tmax')

    update_flgs_tmin_tmax(qa_dly_tmin_tmax_lagrange_inconsist(obs1), 'I')
    
    update_flgs_tmin_tmax(qa_dly_tmin_tmax_mega_inconsist(obs1), 'M')
    
    obs2 = obs.copy()
    obs2['flag_tmin'] = obs1['flag_tmin']
    obs2['flag_tmax'] = obs1['flag_tmax']
    
    return obs2

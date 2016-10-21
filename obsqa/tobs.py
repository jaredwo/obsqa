import pandas as pd

def shift_tmax_tobs_morning(obs, drop_dups=True):
    '''Shift Tmax observations with am observations times back a calendar day
    
    Performs the shift in place. Considers any time-of-observations < 1100 as
    a morning observation. Requires time index and 'tobs_tmax' column variable
    '''
    
    # consider any tobs < 1100 as morning
    # consider 0 as "missing". not sure if this is a valid observation time
    tobs_mask_am = ((obs.tobs_tmax < 1100) & (obs.tobs_tmax != 0))
    
    if tobs_mask_am.any():
        
        idx_name = obs.index.name
        
        obs = obs.copy()
        obs['time'] = obs.index
    
        obs.loc[tobs_mask_am, 'time'] = (obs.loc[tobs_mask_am, 'time'] -
                                         pd.Timedelta(days=1))
        
        if drop_dups:
            # Check for and remove duplicates. A duplicate can occur between the 
            # days that a station switches observation times.
            obs = obs.drop_duplicates(subset=['site_number', 'time'], keep='first')
        
        obs = obs.set_index('time')
        obs.index.name = idx_name
        
    return obs
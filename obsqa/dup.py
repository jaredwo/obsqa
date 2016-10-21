from .util import grt_circle_dist
import difflib
import itertools
import numpy as np
import pandas as pd

class XrefBuilder():
    
    _states = { 'AK': 'Alaska',
                'AL': 'Alabama',
                'AR': 'Arkansas',
                'AS': 'American Samoa',
                'AZ': 'Arizona',
                'CA': 'California',
                'CO': 'Colorado',
                'CT': 'Connecticut',
                'DC': 'District of Columbia',
                'DE': 'Delaware',
                'FL': 'Florida',
                'GA': 'Georgia',
                'GU': 'Guam',
                'HI': 'Hawaii',
                'IA': 'Iowa',
                'ID': 'Idaho',
                'IL': 'Illinois',
                'IN': 'Indiana',
                'KS': 'Kansas',
                'KY': 'Kentucky',
                'LA': 'Louisiana',
                'MA': 'Massachusetts',
                'MD': 'Maryland',
                'ME': 'Maine',
                'MI': 'Michigan',
                'MN': 'Minnesota',
                'MO': 'Missouri',
                'MP': 'Northern Mariana Islands',
                'MS': 'Mississippi',
                'MT': 'Montana',
                'NA': 'National',
                'NC': 'North Carolina',
                'ND': 'North Dakota',
                'NE': 'Nebraska',
                'NH': 'New Hampshire',
                'NJ': 'New Jersey',
                'NM': 'New Mexico',
                'NV': 'Nevada',
                'NY': 'New York',
                'OH': 'Ohio',
                'OK': 'Oklahoma',
                'OR': 'Oregon',
                'PA': 'Pennsylvania',
                'PR': 'Puerto Rico',
                'RI': 'Rhode Island',
                'SC': 'South Carolina',
                'SD': 'South Dakota',
                'TN': 'Tennessee',
                'TX': 'Texas',
                'UT': 'Utah',
                'VA': 'Virginia',
                'VI': 'Virgin Islands',
                'VT': 'Vermont',
                'WA': 'Washington',
                'WI': 'Wisconsin',
                'WV': 'West Virginia',
                'WY': 'Wyoming' }

    _st_abbrv_to_name = {st_abbrv:st_name.upper()
                         for st_abbrv,st_name in _states.items()}
    _st_name_to_abbrv = {st_name.upper():st_abbrv
                         for st_abbrv,st_name in _states.items()}
    
    def __init__(self, obsiof, stns):
        
        self.obsiof = obsiof
        self.stns = stns
    
    def build_xref_snotel(self):
        """Build station id cross-reference for NRCS SNOTEL/SCAN stations
        """
        
        # Missing shefIds that were looked up manually
        # MesoWest_SNV03 match to KALN2 | 14K10S?
        # MesoWest_BSCK1 match to BSKC1 | 19L41S?
        miss_shef = pd.DataFrame({'id_madis':['MesoWest_IDRC2','MesoWest_COUW4',
                                              'MesoWest_SCWI1','MesoWest_XXXC1',
                                              None,None]},
                                 index=['07M27S','10E10S','16A13S','19L06S',
                                        'SC201','SC2124'])
        
        nrcs = self.obsiof.create_obsio_dly_nrcs()
        stns_nrcs = nrcs.stns
        
        #Add corresponding MADIS and GHCND IDs
        stns_nrcs['id_madis'] = "MesoWest_" + stns_nrcs.shefId
        stns_nrcs.id_madis.fillna(miss_shef.id_madis, inplace=True)
        stns_nrcs['id_ghcnd'] = 'USS' + stns_nrcs.station_id.str.zfill(8)
        
        nrcs_xref = stns_nrcs[['station_id','id_ghcnd',
                               'id_madis']].reset_index(drop=True)
        nrcs_xref.rename(columns={'station_id':'id_nrcs'}, inplace=True)
        
        return nrcs_xref
    
    def build_xref_raws(self):
        """Build station id cross-reference for RAWS stations
        """
        
        wrcc = self.obsiof.create_obsio_dly_wrcc_raws()
        stns_wrcc = wrcc.stns
        
        raws_xref = stns_wrcc[['station_id']].reset_index(drop=True)
        raws_xref.rename(columns={'station_id':'id_wrcc'},inplace=True)
        
        #Add corresponding GHCND IDS to xref
        raws_xref['id_ghcnd'] = 'USR' + raws_xref.id_wrcc.str.zfill(8)
        
        #Load MADIS RAWS stations that are in the datastore        
        stns_madis = self.stns[(self.stns.sub_provider=='RAWS') &
                               (self.stns.provider=='MADIS_MESONET')]
        stns_madis = stns_madis.reset_index(drop=True)
        
        #Get the nearest WRCC station to each MADIS station
        def get_nearest_wrcc(stn_madis):
            
            d = grt_circle_dist(stn_madis.longitude, stn_madis.latitude,
                                stns_wrcc.longitude.values,
                                stns_wrcc.latitude.values)
            i = np.argsort(d)[0]
            
            return stns_wrcc.iloc[i].station_id,d[i]
        
        near_wrcc = zip(*stns_madis.apply(get_nearest_wrcc,axis=1))
        stns_madis['near_wrcc_id'],stns_madis['near_wrcc_d'] = near_wrcc
        
        # Extract states from station names
        
        def extract_state_madis_name(name_splits):
            
            try:
                st_abbrv = name_splits[-2]
                self._st_abbrv_to_name[st_abbrv]
                name_splits = name_splits[0:-2]
            except (KeyError, IndexError):
                st_abbrv = np.nan
            
            return ' '.join(name_splits).strip(),st_abbrv
        
        stns_madis['fmtstation_name'],stns_madis['state'] = zip(*stns_madis.station_name.
                                                                str.upper().str.split().
                                                                map(extract_state_madis_name))
        
        def extract_state_wrcc_name(name_splits):
            
            try:
            
                st_abbrv = self._st_name_to_abbrv[name_splits[-1]]
                name_splits = name_splits[0:-1]
            
            except (KeyError, IndexError):
                
                try:
                    
                    st_abbrv = self._st_name_to_abbrv[' '.join(name_splits[-2:])]
                    name_splits = name_splits[0:-2]
                    
                except (KeyError, IndexError):
                    
                    st_abbrv = np.nan
                    
            return ' '.join(name_splits).strip(),st_abbrv
        
        stns_wrcc['fmtstation_name'],stns_wrcc['state'] = zip(*stns_wrcc.station_name.
                                                              str.upper().str.split().
                                                              map(extract_state_wrcc_name))
        
        # Check if nearest WRCC station has a close name match to the MADIS
        # station.
        def is_close_name_match(stn):
            
            try:
                
                difflib.get_close_matches(stn.fmtstation_name,
                                          [stns_wrcc.loc[stn.near_wrcc_id].
                                           fmtstation_name])[0]
                is_close = True
        
            except IndexError:
                
                is_close = False
                
            return is_close
        
        stns_madis['has_name_match'] = stns_madis.apply(is_close_name_match,axis=1)
        
        # A MADIS RAWS station is considered a duplicate of a WRCC RAWS station
        # if it is within 1-km of the WRCC RAWS station or within 40-km and has
        # a close name match.
        mask_match = ((stns_madis.near_wrcc_d <= 1.0) |
                      ((stns_madis.near_wrcc_d <= 40.0) & 
                       (stns_madis.has_name_match)))
        
        stns_madis_match = stns_madis.loc[mask_match,['station_id','near_wrcc_id']]
        stns_madis_match = stns_madis_match[stns_madis_match.station_id != 'RAWS_']
        stns_madis_match.rename(columns={'station_id':'id_madis',
                                         'near_wrcc_id':'id_wrcc'}, inplace=True)
        stns_madis_match = stns_madis_match.drop_duplicates()
        
        raws_xref = raws_xref.merge(stns_madis_match, how='left',
                                    on='id_wrcc').drop_duplicates()
        
        return raws_xref
    
    def build_xref_wban(self):
        """Build station id cross-reference for stations with WBAN ID and MADIS METAR
        """
                
        # Get ISD-Lite stations that have a WBAN ID
        isd = self.obsiof.create_obsio_dly_isdlite()
        stns_isd = isd.stns.copy()
        stns_isd['id_wban'] = stns_isd.station_id.str.split('-',expand=True)[1]
        stns_isd.loc[stns_isd.id_wban=='99999','id_wban'] = np.nan
        stns_isd.dropna(subset=['id_wban'], inplace=True)
        # Add corresponding GHCND IDS
        stns_isd['id_ghcnd'] = 'USW' + stns_isd.id_wban.str.zfill(8)
        
        #Get ACIS station metadata for crossing referencing WBAN and MADIS METAR
        #ICAO IDS
        acis = self.obsiof.create_obsio_dly_acis()
        stns_acis = acis.stns.copy()
        
        for sidcol in stns_acis.columns[stns_acis.columns.str.startswith('sid')]:
            try:
                stns_acis[sidcol] = stns_acis[sidcol].str.split(expand=True)[0]
            except AttributeError:
                continue
        stns_acis = stns_acis[(~stns_acis.sid_icao.isnull()) & (~stns_acis.sid_wban.isnull())]
        
        #Load MADIS METAR stations that are in the datastore
        stns_madis = self.stns[self.stns.provider=='MADIS_METAR']
        stns_madis = stns_madis.reset_index(drop=True)
        
        #Cross reference WBAN with ICAO
        stns_madis = stns_madis[['station_id']].merge(stns_acis[['sid_wban','sid_icao']],
                                                      how='inner', left_on='station_id',
                                                      right_on='sid_icao')[['station_id','sid_wban']]
        stns_madis.rename(columns={'station_id':'id_madis','sid_wban':'id_wban'},
                          inplace=True)
        
        wban_xref = stns_isd[['id_wban','id_ghcnd']].reset_index()
        wban_xref.rename(columns={'station_id':'id_isdlite'},inplace=True)
        
        wban_xref = wban_xref.merge(stns_madis,how='left',on='id_wban')
        wban_xref = wban_xref[['id_ghcnd','id_isdlite','id_madis']]
        wban_xref = wban_xref.drop_duplicates()
        
        return wban_xref

    def build_xref_crn(self):
        """Build station id cross-reference for CRN stations
        """
        
        # Get CRN GHCND stations
        ghcnd = self.obsiof.create_obsio_dly_ghcnd()
        stns_ghcnd = ghcnd.stns[ghcnd.stns.hcn_crn_flag == 'CRN'].copy()
        
        # GET ISD stations and cross reference with GHCND based on WBAN ID
        isd = self.obsiof.create_obsio_dly_isdlite()
        stns_isd = isd.stns.copy()
        stns_isd['id_wban'] = stns_isd.station_id.str.split('-',expand=True)[1]
        stns_isd.loc[stns_isd.id_wban=='99999','id_wban'] = np.nan
        stns_isd.dropna(subset=['id_wban'], inplace=True)
        #Add corresponding GHCND IDS
        stns_isd['id_ghcnd'] = 'USW' + stns_isd.id_wban.str.zfill(8)
        stns_isd.rename(columns={'station_id':'id_isdlite'}, inplace=True)
        
        xref_crn = stns_ghcnd.merge(stns_isd[['id_ghcnd','id_isdlite']], 
                                    how='left',left_on='station_id',
                                    right_on='id_ghcnd')[['station_id','id_isdlite']]
        xref_crn.rename(columns={'station_id':'id_ghcnd'}, inplace=True)
        
        # Get ACIS station metadata for crossing referencing GHCND CRN IDS and
        # MADIS NWSLI CRN IDS
        acis = self.obsiof.create_obsio_dly_acis()
        stns_acis = acis.stns.copy()
        
        for sidcol in stns_acis.columns[stns_acis.columns.str.startswith('sid')]:
            try:
                stns_acis[sidcol] = stns_acis[sidcol].str.split(expand=True)[0]
            except AttributeError:
                continue
        stns_acis = stns_acis[(~stns_acis.sid_nwsli.isnull()) & (~stns_acis.sid_ghcn.isnull())]
        
        #Load MADIS CRN stations that are in the datastore
        stns_madis = self.stns[(self.stns.provider=='MADIS_COOP') &
                               (self.stns.sub_provider=='CRN')]
        stns_madis = stns_madis.reset_index(drop=True)
        stns_madis['station_id'] = stns_madis.station_id.str.split('_').str[-1]
        
        #Cross reference NWSLI with GHCND
        stns_madis = stns_madis[['station_id']].merge(stns_acis[['sid_nwsli','sid_ghcn']],
                                                      how='inner', left_on='station_id',
                                                      right_on='sid_nwsli')[['station_id','sid_ghcn']]
        stns_madis.rename(columns={'station_id':'id_madis','sid_ghcn':'id_ghcnd'},
                          inplace=True)
        stns_madis['id_madis'] = 'CRN_' + stns_madis.id_madis
        
        xref_crn = xref_crn.merge(stns_madis,on='id_ghcnd')
        
        return xref_crn

    def build_xref_location(self):
        """Build site number cross-reference for stations in duplicate locations
        """
        
        stns = self.stns.copy()
        stns.dropna(how='any', subset=['longitude', 'latitude', 'elevation'],
                    inplace=True)
        
        #Consider duplicates if within 0.001 degree and 10 m in elevation
        stns['rnd_longitude'] = stns.longitude.round(3)
        stns['rnd_latitude'] = stns.latitude.round(3)
        stns['rnd_elevation'] = (stns.elevation/10.0).round()*10.0
        
        stns['flr_longitude'] = np.floor((stns.longitude*1000.0).values)/1000.0
        stns['flr_latitude'] = np.floor((stns.latitude*1000.0).values)/1000.0
        stns['flr_elevation'] = np.floor((stns.elevation/10.0).values)*10.0
        
        lon_names = ['rnd_longitude','flr_longitude']
        lat_names = ['rnd_latitude','flr_latitude']
        elev_names = ['rnd_elevation','flr_elevation']
        
        max_dupgrp = 0
        
        dup_stns_all = []
        
        for loc_names in itertools.product(lon_names, lat_names, elev_names):
            
            dup_stns = stns[stns.duplicated(subset=loc_names,keep=False)].copy()
            dup_stns['dup_grp'] = np.arange(len(dup_stns)) + max_dupgrp
            dup_stns['dup_grp'] = dup_stns.groupby(loc_names)['dup_grp'].transform(np.min)
            dup_stns.set_index('dup_grp',inplace=True)
            
            i_prvdr = np.arange(dup_stns.groupby(dup_stns.index).size().max())
            i_prvdr = np.array(['%.2d'%i for i in i_prvdr])
            i_prvdr[0] = ''
            grped = dup_stns.groupby([dup_stns.index,
                                      dup_stns.provider])['provider']
            prvdr_rename = lambda x: x + i_prvdr[0:x.size]
            dup_stns['provider'] = grped.transform(prvdr_rename)
            
            max_dupgrp = dup_stns.index.max() + 1
            dup_stns_all.append(dup_stns[['site_number','station_id', 'provider']])
        
        dup_stns_all = pd.concat(dup_stns_all)
        
        dups_pivot = dup_stns_all[['site_number',
                                   'provider']].pivot(columns='provider',
                                                      values='site_number')
        dups_pivot = dups_pivot.drop_duplicates()

        provider_priority = ['WRCC','NRCS','GHCND','ACIS','ISD-Lite','MADIS_METAR',
                             'MADIS_MESONET','MADIS_COOP','MADIS_SAO','LOGTAG']

        cols_ordered = []
        
        for a_prvdr in provider_priority:
            
            cols_ordered.extend(dups_pivot.columns[dups_pivot.columns.str.startswith(a_prvdr)])
            
        dups_pivot = dups_pivot[cols_ordered].reset_index(drop=True)
        dups_pivot.columns.name = ''
        
        return dups_pivot

class Deduper():
    
    def __init__(self, xrefs, xref_loc=None):
        
        self.xrefs = xrefs
        self.xref_loc = xref_loc

    def dedup(self, obs):
        
        ids_drop = []
        
        for xref in self.xrefs:
                                    
            stns_in = xref.isin(obs.station_id.unique()).values
            in_cumsum = np.cumsum(stns_in, axis=1)
            ids_drop.extend(xref.values[in_cumsum>1])
                        
        obs_dedup = obs[~obs.station_id.isin(ids_drop)]
        
        if self.xref_loc is not None:
                        
            stns_in = self.xref_loc.isin(obs_dedup.site_number.unique()).values
            in_cumsum = np.cumsum(stns_in, axis=1)
            sns_drop = self.xref_loc.values[in_cumsum>1].astype(np.int)       
            obs_dedup = obs_dedup[~obs_dedup.site_number.isin(sns_drop)]
            
        return obs_dedup

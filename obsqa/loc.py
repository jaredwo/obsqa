from time import sleep
from urllib2 import HTTPError
import json
import numpy as np
import urllib
import urllib2

def _get_elev_usgs(lon, lat, maxtries):
    """Get elev value from USGS NED 1/3 arc-sec DEM.

    http://ned.usgs.gov/epqs/
    """

    URL_USGS_NED = 'http://ned.usgs.gov/epqs/pqs.php'
    USGS_NED_NODATA = -1000000

    # url GET args
    values = {'x': lon,
              'y': lat,
              'units': 'Meters',
              'output': 'json'}

    data = urllib.urlencode(values)

    req = urllib2.Request(URL_USGS_NED, data)
                
    ntries = 0
    
    while 1:
        
        try:
            
            response = urllib2.urlopen(req)
            break

        except HTTPError:
            
            ntries += 1
        
            if ntries >= maxtries:
        
                raise
        
            sleep(1)
            
    json_response = json.loads(response.read())
    elev = np.float(json_response['USGS_Elevation_Point_Query_Service']
                    ['Elevation_Query']['Elevation'])

    if elev == USGS_NED_NODATA:

        elev = np.nan

    return elev

def _get_elev_geonames(lon, lat, usrname_geonames, maxtries):
    """Get elev value from geonames web sevice (SRTM or ASTER)
    """

    URL_GEONAMES_SRTM = 'http://api.geonames.org/srtm3'
    URL_GEONAMES_ASTER = 'http://api.geonames.org/astergdem'

    url = URL_GEONAMES_SRTM

    while 1:
        # ?lat=50.01&lng=10.2&username=demo
        # url GET args
        values = {'lat': lat, 'lng': lon, 'username': usrname_geonames}

        # encode the GET arguments
        data = urllib.urlencode(values)

        # make the URL into a qualified GET statement
        get_url = "".join([url, "?", data])

        req = urllib2.Request(url=get_url)
        
        ntries = 0
        
        while 1:
            
            try:
                
                response = urllib2.urlopen(req)
                break

            except HTTPError:
                
                ntries += 1
            
                if ntries >= maxtries:
            
                    raise
            
                sleep(1)
        
        elev = float(response.read().strip())

        if elev == -32768.0 and url == URL_GEONAMES_SRTM:
            # Try ASTER instead
            url = URL_GEONAMES_ASTER
        else:
            break

    if elev == -32768.0 or elev == -9999.0:
        elev = np.nan

    return elev

def get_elevation(lon, lat, usrname_geonames=None, maxtries=3):

    elev = _get_elev_usgs(lon, lat, maxtries)

    if np.isnan(elev) and usrname_geonames is not None:

        elev = _get_elev_geonames(lon, lat, usrname_geonames, maxtries)

    return elev

def get_invalid_loc(stns):
    
    elev_dif = (stns['elevation'] - stns['elevation_dem']).abs()
    
    id_locqa_fail = stns.index[((stns.elevation.isnull()) |
                                (elev_dif.isnull()) |
                                (elev_dif >= 200))]
    return id_locqa_fail

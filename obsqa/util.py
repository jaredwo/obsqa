import numpy as np

_RADIAN_CONVERSION_FACTOR = 0.017453292519943295 #pi/180
_AVG_EARTH_RADIUS_KM = 6371.009 #Mean earth radius as defined by IUGG

def grt_circle_dist(lon1,lat1,lon2,lat2):
        '''Calculate great circle distance according to the haversine formula
        
        See http://en.wikipedia.org/wiki/Great-circle_distance
        '''
        #convert to radians
        lat1rad = lat1 * _RADIAN_CONVERSION_FACTOR
        lat2rad = lat2 * _RADIAN_CONVERSION_FACTOR
        lon1rad = lon1 * _RADIAN_CONVERSION_FACTOR
        lon2rad = lon2 * _RADIAN_CONVERSION_FACTOR
        deltaLat = lat1rad - lat2rad
        deltaLon = lon1rad - lon2rad
        centralangle = 2 * np.arcsin(np.sqrt((np.sin (deltaLat/2))**2 +
                                             np.cos(lat1rad) * np.cos(lat2rad)
                                             * (np.sin(deltaLon/2))**2))
        #average radius of earth times central angle, result in kilometers
        #distDeg = centralangle/RADIAN_CONVERSION_FACTOR
        distKm = _AVG_EARTH_RADIUS_KM * centralangle 
        return distKm
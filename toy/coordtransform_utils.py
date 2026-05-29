# -*- coding: utf-8 -*-
from math import sin, cos, sqrt, fabs, atan2
from math import pi as PI


# define ellipsoid
a = 6378245.0
f = 1 / 298.3
b = a * (1 - f)
ee = 1 - (b * b) / (a * a)


def out_of_china(lng, lat):
    """Check whether lng and lat are out of China

    Arguments:
        lng {float} -- longitude
        lat {float} -- latitude

    Returns:
        bool -- True or False
    """
    return not (72.004 <= lng <= 137.8347 and 0.8293 <= lat <= 55.8271)


def transform_lat(x, y):
    """Transform latitude

    Arguments:
        x {float} -- x coordinate
        y {float} -- y coordinate

    Returns:
        float -- transformed latitude
    """
    ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * \
        y * y + 0.1 * x * y + 0.2 * sqrt(fabs(x))
    ret = ret + (20.0 * sin(6.0 * x * PI) + 20.0 *
                 sin(2.0 * x * PI)) * 2.0 / 3.0
    ret = ret + (20.0 * sin(y * PI) + 40.0 * sin(y / 3.0 * PI)) * 2.0 / 3.0
    ret = ret + (160.0 * sin(y / 12.0 * PI) + 320.0 *
                 sin(y * PI / 30.0)) * 2.0 / 3.0
    return ret


def transform_lon(x, y):
    """Transform longitude

    Arguments:
        x {float} -- x coordinate
        y {float} -- y coordinate

    Returns:
        float -- transformed longitude
    """
    ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * sqrt(fabs(x))
    ret = ret + (20.0 * sin(6.0 * x * PI) + 20.0 *
                 sin(2.0 * x * PI)) * 2.0 / 3.0
    ret = ret + (20.0 * sin(x * PI) + 40.0 * sin(x / 3.0 * PI)) * 2.0 / 3.0
    ret = ret + (150.0 * sin(x / 12.0 * PI) + 300.0 *
                 sin(x * PI / 30.0)) * 2.0 / 3.0
    return ret


def wgs_to_gcj(wgs_lon, wgs_lat):
    """Convert WGS84 coordinates to GCJ02 coordinates

    Arguments:
        wgs_lon {float} -- longitude in WGS84
        wgs_lat {float} -- latitude in WGS84

    Returns:
        tuple -- GCJ02 coordinates (longitude, latitude)
    """

    if out_of_china(wgs_lon, wgs_lat):
        return wgs_lon, wgs_lat
    d_lat = transform_lat(wgs_lon - 105.0, wgs_lat - 35.0)
    d_lon = transform_lon(wgs_lon - 105.0, wgs_lat - 35.0)
    rad_lat = wgs_lat / 180.0 * PI
    magic = sin(rad_lat)
    magic = 1 - ee * magic * magic
    sqrt_magic = sqrt(magic)
    d_lat = (d_lat * 180.0) / ((a * (1 - ee)) / (magic * sqrt_magic) * PI)
    d_lon = (d_lon * 180.0) / (a / sqrt_magic * cos(rad_lat) * PI)
    gcj_lat = wgs_lat + d_lat
    gcj_lon = wgs_lon + d_lon
    return (gcj_lon, gcj_lat)


def gcj_to_wgs(gcj_lon, gcj_lat):
    """Convert GCJ02 coordinates to WGS84 coordinates

    Arguments:
        gcj_lon {float} -- longitude in GCJ02
        gcj_lat {float} -- latitude in GCJ02

    Returns:
        tuple -- WGS84 coordinates (longitude, latitude)
    """
    g0 = (gcj_lon, gcj_lat)
    w0 = g0
    g1 = wgs_to_gcj(w0[0], w0[1])
    # w1 = w0 - (g1 - g0)
    w1 = tuple(map(lambda x: x[0]-(x[1]-x[2]), zip(w0, g1, g0)))
    # delta = w1 - w0
    delta = tuple(map(lambda x: x[0] - x[1], zip(w1, w0)))
    while (abs(delta[0]) >= 1e-6 or abs(delta[1]) >= 1e-6):
        w0 = w1
        g1 = wgs_to_gcj(w0[0], w0[1])
        # w1 = w0 - (g1 - g0)
        w1 = tuple(map(lambda x: x[0]-(x[1]-x[2]), zip(w0, g1, g0)))
        # delta = w1 - w0
        delta = tuple(map(lambda x: x[0] - x[1], zip(w1, w0)))
    return w1


def gcj_to_bd(gcj_lon, gcj_lat):
    """Convert GCJ02 coordinates to BD09 coordinates

    Arguments:
        gcj_lon {float} -- longitude in GCJ02
        gcj_lat {float} -- latitude in GCJ02

    Returns:
        tuple -- BD09 coordinates (longitude, latitude)
    """
    z = sqrt(gcj_lon * gcj_lon + gcj_lat * gcj_lat) + \
        0.00002 * sin(gcj_lat * PI * 3000.0 / 180.0)
    theta = atan2(gcj_lat, gcj_lon) + 0.000003 * \
        cos(gcj_lon * PI * 3000.0 / 180.0)
    bd_lon = z * cos(theta) + 0.0065
    bd_lat = z * sin(theta) + 0.006
    return (bd_lon, bd_lat)


def bd_to_gcj(bd_lon, bd_lat):
    """Convert BD09 coordinates to GCJ02 coordinates

    Arguments:
        bd_lon {float} -- longitude in BD09
        bd_lat {float} -- latitude in BD09

    Returns:
        tuple -- GCJ02 coordinates (longitude, latitude)
    """
    x = bd_lon - 0.0065
    y = bd_lat - 0.006
    z = sqrt(x * x + y * y) - 0.00002 * sin(y * PI * 3000.0 / 180.0)
    theta = atan2(y, x) - 0.000003 * cos(x * PI * 3000.0 / 180.0)
    gcj_lon = z * cos(theta)
    gcj_lat = z * sin(theta)
    return (gcj_lon, gcj_lat)


def wgs_to_bd(wgs_lon, wgs_lat):
    """Convert WGS84 coordinates to BD09 coordinates

    Arguments:
        wgs_lon {float} -- longitude in WGS84
        wgs_lat {float} -- latitude in WGS84

    Returns:
        tuple -- BD09 coordinates (longitude, latitude)
    """
    gcj = wgs_to_gcj(wgs_lon, wgs_lat)
    return gcj_to_bd(gcj[0], gcj[1])


def bd_to_wgs(bd_lon, bd_lat):
    """Convert BD09 coordinates to WGS84 coordinates

    Arguments:
        bd_lon {float} -- longitude in BD09
        bd_lat {float} -- latitude in BD09

    Returns:
        tuple -- WGS84 coordinates (longitude, latitude)
    """
    gcj = bd_to_gcj(bd_lon, bd_lat)
    return gcj_to_wgs(gcj[0], gcj[1])


class Transform():
    """Coordinate transformation class for converting between different coordinate systems"""

    def transform_lat(self, x, y):
        """Transform latitude

        Arguments:
            x {float} -- x coordinate
            y {float} -- y coordinate

        Returns:
            float -- transformed latitude
        """
        return transform_lat(x, y)

    def transform_lon(self, x, y):
        """Transform longitude

        Arguments:
            x {float} -- x coordinate
            y {float} -- y coordinate

        Returns:
            float -- transformed longitude
        """
        return transform_lon(x, y)

    def wgs_to_gcj(self, wgs_lon, wgs_lat):
        """Convert WGS84 coordinates to GCJ02 coordinates

        Arguments:
            wgs_lon {float} -- longitude in WGS84
            wgs_lat {float} -- latitude in WGS84

        Returns:
            tuple -- GCJ02 coordinates (longitude, latitude)
        """
        return wgs_to_gcj(wgs_lon, wgs_lat)

    def gcj_to_wgs(self, gcj_lon, gcj_lat):
        """Convert GCJ02 coordinates to WGS84 coordinates

        Arguments:
            gcj_lon {float} -- longitude in GCJ02
            gcj_lat {float} -- latitude in GCJ02

        Returns:
            tuple -- WGS84 coordinates (longitude, latitude)
        """
        return gcj_to_wgs(gcj_lon, gcj_lat)

    def gcj_to_bd(self, gcj_lon, gcj_lat):
        """Convert GCJ02 coordinates to BD09 coordinates

        Arguments:
            gcj_lon {float} -- longitude in GCJ02
            gcj_lat {float} -- latitude in GCJ02

        Returns:
            tuple -- BD09 coordinates (longitude, latitude)
        """
        return gcj_to_bd(gcj_lon, gcj_lat)

    def bd_to_gcj(self, bd_lon, bd_lat):
        """Convert BD09 coordinates to GCJ02 coordinates

        Arguments:
            bd_lon {float} -- longitude in BD09
            bd_lat {float} -- latitude in BD09

        Returns:
            tuple -- GCJ02 coordinates (longitude, latitude)
        """
        return bd_to_gcj(bd_lon, bd_lat)

    def wgs_to_bd(self, wgs_lon, wgs_lat):
        """Convert WGS84 coordinates to BD09 coordinates

        Arguments:
            wgs_lon {float} -- longitude in WGS84
            wgs_lat {float} -- latitude in WGS84

        Returns:
            tuple -- BD09 coordinates (longitude, latitude)
        """
        return wgs_to_bd(wgs_lon, wgs_lat)

    def bd_to_wgs(self, bd_lon, bd_lat):
        """Convert BD09 coordinates to WGS84 coordinates

        Arguments:
            bd_lon {float} -- longitude in BD09
            bd_lat {float} -- latitude in BD09

        Returns:
            tuple -- WGS84 coordinates (longitude, latitude)
        """
        return bd_to_wgs(bd_lon, bd_lat)

import numpy as np
def orientation_3D_to_rgb(hsv, interp_belt = 20/180*np.pi, sat_factor = 1):
    """
    Convert hsv values to rgb.
    Parameters
    ----------
    hsv : (..., 3) array-like
       [h, s, v] is refered to [hue, saturation(inclination), value]
       h and v are assumed to be in range [0, 1]
       s is assumed to be in range of [0, pi]
    Returns
    -------
    rgb : (..., 3) ndarray
       Colors converted to RGB values in range [0, 1]
    """
    hsv = np.asarray(hsv)

    # check length of the last dimension, should be _some_ sort of rgb
    if hsv.shape[-1] != 3:
        raise ValueError("Last dimension of input array must be 3; "
                         "shape {shp} was found.".format(shp=hsv.shape))

    in_shape = hsv.shape
    hsv = np.array(
        hsv, copy=False,
        dtype=np.promote_types(hsv.dtype, np.float32),  # Don't work on ints.
        ndmin=2,  # In case input was 1D.
    )
    
    s_sign = np.sign(np.pi/2-hsv[..., 1])
    
    theta_merge_1 = (hsv[..., 1] - (np.pi/2 - interp_belt))/(2*interp_belt)
    theta_merge_2 = 1 - theta_merge_1
    
    scaling_factor = np.zeros_like(hsv[..., 1])
    idx_scale = theta_merge_1 <= 0
    scaling_factor[idx_scale] = hsv[idx_scale, 1] / (np.pi/2 - interp_belt) 
    
    idx_scale = np.logical_and(theta_merge_1>0, theta_merge_2>0)
    scaling_factor[idx_scale] = 1 
    
    idx_scale = theta_merge_2 <= 0
    scaling_factor[idx_scale] = (np.pi-hsv[idx_scale, 1]) / (np.pi/2 - interp_belt) 
    h = hsv[..., 0]
    s = np.sin(scaling_factor**sat_factor * np.pi/2)
    v = hsv[..., 2]
    
    r = np.empty_like(h)
    g = np.empty_like(h)
    b = np.empty_like(h)

    i = (h * 6.0).astype(int)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))

    idx = np.logical_and(i % 6 == 0, s_sign>0)
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 1, s_sign>0)
    r[idx] = q[idx]
    g[idx] = t[idx]
    b[idx] = q[idx]

    idx = np.logical_and(i == 2, s_sign>0)
    r[idx] = t[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = np.logical_and(i == 3, s_sign>0)
    r[idx] = q[idx]
    g[idx] = q[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 4, s_sign>0)
    r[idx] = p[idx]
    g[idx] = t[idx]
    b[idx] = v[idx]

    idx = np.logical_and(i == 5, s_sign>0)
    r[idx] = t[idx]
    g[idx] = q[idx]
    b[idx] = q[idx]
    
    # the other hemisphere
    
    idx = np.logical_and(i == 3, s_sign<0)
    r[idx] = v[idx]
    g[idx] = p[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 4, s_sign<0)
    r[idx] = q[idx]
    g[idx] = t[idx]
    b[idx] = q[idx]

    idx = np.logical_and(i == 5, s_sign<0)
    r[idx] = t[idx]
    g[idx] = v[idx]
    b[idx] = p[idx]

    idx = np.logical_and(i % 6 == 0, s_sign<0)
    r[idx] = q[idx]
    g[idx] = q[idx]
    b[idx] = t[idx]

    idx = np.logical_and(i == 1, s_sign<0)
    r[idx] = p[idx]
    g[idx] = t[idx]
    b[idx] = v[idx]

    idx = np.logical_and(i == 2, s_sign<0)
    r[idx] = t[idx]
    g[idx] = q[idx]
    b[idx] = q[idx]

    # inclination color blending
    idx_blend = np.logical_and(i % 6 == 0, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = v[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = p[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    
    idx_blend = np.logical_and(i == 1, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + p[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + v[idx_blend]*theta_merge_1[idx_blend]
    
    idx_blend = np.logical_and(i == 2, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = v[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = p[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    
    idx_blend = np.logical_and(i == 3, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + v[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + p[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    
    idx_blend = np.logical_and(i == 4, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = p[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = v[idx_blend]*theta_merge_2[idx_blend] + q[idx_blend]*theta_merge_1[idx_blend]
    
    idx_blend = np.logical_and(i == 5, np.logical_and(theta_merge_1 > 0, theta_merge_2 > 0))
    r[idx_blend] = t[idx_blend]*theta_merge_2[idx_blend] + t[idx_blend]*theta_merge_1[idx_blend]
    g[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + v[idx_blend]*theta_merge_1[idx_blend]
    b[idx_blend] = q[idx_blend]*theta_merge_2[idx_blend] + p[idx_blend]*theta_merge_1[idx_blend]

    idx = s == 0
    r[idx] = v[idx]
    g[idx] = v[idx]
    b[idx] = v[idx]

    rgb = np.stack([r, g, b], axis=-1)

    return rgb.reshape(in_shape)
import cv2
import numpy as np


def extent(image):
    
    ''' Area / bounding box area. '''
    return (image == 255).sum() / float(image.size)


def chen_stroke_width(image, mask=None):
    ''' STW transform '''
 
    assert image.ndim == 2
    
    if mask is None:
        mask = np.ones_like(image).astype(np.bool)
    else:
        mask = (mask == 255)
                                
    # Foreground pixels outside borders
    padded = np.pad(image, (1, 1), mode='constant', constant_values=0)
    dist = cv2.distanceTransform(padded, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    dist = np.round(dist)
    max_stroke = dist.max()
    if max_stroke == 0.0:
        return None, np.inf, np.inf
        
    # 8-neighborhood
    N8 = lambda (y, x) : (np.array( \
                             [y - 1, y - 1, y - 1,
                              y,     y,
                              y + 1, y + 1, y + 1],
                              np.int16),
                          np.array(    
                             [x - 1, x,     x + 1,
                              x - 1, x + 1,
                              x - 1, x,     x + 1],
                              np.int16))
    # Lookup
    fy, fx = np.where(padded== 255)
    lookup = {}
    for y, x in zip(fy, fx):
        Ny, Nx = N8((y, x))
        values = dist[Ny, Nx]
        ref_value = dist[y, x]
        mask_values = np.logical_and(values < ref_value, values > 0)
        lookup[(y, x)] = (Ny[mask_values], Nx[mask_values])
        
    # Down-hill
    for stroke in range(max_stroke, 0, -1):
        Sy, Sx = np.where(dist == stroke)
        neighbors = set([n for p in zip(Sy, Sx) for n in zip(* lookup[p])])
        while neighbors:
            Ny, Nx = zip(* neighbors)
            dist[Ny, Nx] = stroke
            neighbors = set([n for p in neighbors for n in zip(* lookup[p])])
            
    dist = dist[1 : -1, 1 : -1]
    values = dist[np.logical_and(image == 255, mask)]
    var = values.std() / values.mean()
#    thickness = 2 * values.max()
    thickness = 2 * values.mean()
    
    return dist, var, thickness
            
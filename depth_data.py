"""
Depth Data

@author: Morgen Malinoski
"""

import pyrealsense2 as rs

class DepthFramePickleable:
    # Save  all possible distances which could be used
    def __init__(self, depth, render_image):
        # Bounds from image
        rows, cols, _ = render_image.shape[:3]
        low_bound_x = 0
        upper_bound_x = cols - 1
        low_bound_y = 0
        upper_bound_y = rows - 1
        
        self.distances = {}
        for x in range(low_bound_x, upper_bound_x):
            for y in range(low_bound_y, upper_bound_y):
                self.distances[x,y] = (depth.get_distance(x, y))
                
    def get_distance(self, x, y):
        return self.distances[x,y]
        
class IntrinsicsPickleable(rs.pyrealsense2.intrinsics):
    def __init__(self, intrinsics):
        super().__init__()
        self.coeffs = intrinsics.coeffs
        self.fx = intrinsics.fx
        self.fy = intrinsics.fy
        self.height = intrinsics.height
        self.model = intrinsics.model
        self.ppx = intrinsics.ppx
        self.ppy = intrinsics.ppy
        self.width = intrinsics.width
        
    def __setstate__(self, d):
        self.coeffs = d[0]
        self.fx = d[1]
        self.fy = d[2]
        self.height = d[3]
        self.model = d[4]
        self.ppx = d[5]
        self.ppy = d[6]
        self.width = d[7]
        
    def __getstate__(self):
        return [self.coeffs, self.fx, self.fy, self.height, self.model, self.ppx, self.ppy, self.width]

# Single loaded frame for recording or playback


class DataFrame:
    def __init__(self, color_image, skeletons, depth, depth_intrinsic):
        self.color_image = color_image
        self.skeletons = skeletons
        self.depth = depth
        self.depth_intrinsic = depth_intrinsic
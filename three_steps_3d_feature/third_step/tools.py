import numpy as np
from argparse import Namespace
import quaternion
from habitat.utils.geometry_utils import quaternion_to_list
from scipy.spatial.transform import Rotation as R

class Pos2Map:
    
    def __init__(self, x, y, heading) -> None:
        self.x = x
        self.y = y
        self.heading = heading
        
        
class Pos2World:
    
    def __init__(self, x, y, z, heading) -> None:
        self.x = x
        self.y = y
        self.z = z
        self.heading = heading
        
        
class Geometry_Tools:

    def __init__(self, image_resolution, fov, camera_height) -> None:
        self._camera_matrix = self._parse_camera_matrix(*image_resolution, fov)
        self._camera_height = camera_height
        
    def _parse_r_matrix(self, ax_, angle):
        ax = ax_ / np.linalg.norm(ax_)
        if np.abs(angle) > 0.001:
            S_hat = np.array(
                [[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]],
                dtype=np.float32)
            R = np.eye(3) + np.sin(angle)*S_hat + \
                (1-np.cos(angle))*(np.linalg.matrix_power(S_hat, 2))
        else:
            R = np.eye(3)
        return R
    
    def _parse_camera_matrix(self, width, height, fov):
        xc = (width-1.) / 2.
        zc = (height-1.) / 2.
        f = (width / 2.) / np.tan(np.deg2rad(fov / 2.))
        camera_matrix = {'xc':xc, 'zc':zc, 'f':f}
        camera_matrix = Namespace(**camera_matrix)
        return camera_matrix
    
    def transformation_robot2world(self, goal2robot2world:list, pos2world:Pos2World) -> list:
        """transform the point relative to robot to the point relative to th world

        Args:
            goal2robot2world (list): = [u, v] u is in the right relative to the robot and v is in the forward of th robot
                                        robot first moves according to the v frame and finally u frame
            pos2world (Pos2World): _description_

        Returns:
            list: _description_
        """        
        u, v = goal2robot2world
        x0, y0, z0 = pos2world.x, pos2world.z, pos2world.z
        x1 = x0 + v * np.cos(pos2world.heading + np.pi/2)
        z1 = -(-z0 + v * np.sin(pos2world.heading + np.pi/2))
        x2 = x1 + u * np.cos(pos2world.heading + np.pi/2 - np.pi/2)
        z2 = -(-z1 + u * np.sin(pos2world.heading + np.pi/2 - np.pi/2))
        return [x2, y0, z2]
    
    def transformation_robotbase2map(self, point_clouds_2robotbase:np.array, pos2map:Pos2Map, resolution_meter2pixel) -> np.array:
        """Mapping the points with the robot base as the coordinate system to the map coordinate system

        Args:
            point_clouds_2robotbase (np.array): 
            pos2map (Pos2Map): 
            resolution_meter2pixel (_type_): 

        Returns:
            np.array: point_clouds_2map
        """        
        R = self._parse_r_matrix([0.,0.,1.], angle=pos2map.heading-np.pi/2.)
        point_clouds_2map = np.matmul(point_clouds_2robotbase.reshape(-1,3), R.T).reshape(point_clouds_2robotbase.shape)
        point_clouds_2map[:,:,0] = point_clouds_2map[:,:,0] + pos2map.x * resolution_meter2pixel
        point_clouds_2map[:,:,1] = point_clouds_2map[:,:,1] + pos2map.y * resolution_meter2pixel
        return point_clouds_2map
    
    def transformation_robotcamera2base(self, point_clouds:np.array) -> np.array:
        """Mapping the points with the robot camera as the coordinate system to the robot base coordinate system

        Args:
            point_clouds (np.array): In shape (width, height, 3); 
                                     point_clouds[0] means X cooardinate point_clouds[1] means Y cooardinate point_clouds[2] means Z cooardinate

        Returns:
            np.array: Array of point clouds relative to the robot base coordinate system; In shape (width, height, 3)
        """      
        point_clouds[...,2] = point_clouds[...,2] + self._camera_height  
        return point_clouds
    
    def transformation_camera2robotcamera(self, depth_img:np.array) -> np.array:
        """Mapping the points on the depth map to points with the robot camera as the coordinate system

        Args:
            depth_img (np.array): In shape (width, height, 1); The unit of pixel value is 10 meters

        Returns:
            np.array: Array of point clouds relative to the robot camera coordinate system; In shape (width, height, 3)
        """ 
        x, z = np.meshgrid(np.arange(depth_img.shape[-2]),
                            np.arange(depth_img.shape[-3]-1, -1, -1))
        for _ in range(depth_img.ndim-3):
            x = np.expand_dims(x, axis=0)
            z = np.expand_dims(z, axis=0)
        X = (x-self._camera_matrix.xc) * depth_img[:,:,0] / self._camera_matrix.f
#        print(depth_img)
        Z = (z-self._camera_matrix.zc) * depth_img[:,:,0] / self._camera_matrix.f
        pc = np.concatenate((X[...,np.newaxis], depth_img, Z[...,np.newaxis]), axis=2)
        return pc
    
    def transformation_pointcloud2occupiedmap(self, point_clouds_2map:np.array, map_size, z_bins:list, resolution_meter2pixel, free_index, occupied_index) -> np.array :
        """project the point cloud relative to the map coordinate system to the top view

        Args:
            point_clouds_2map (np.array): 
            map_size (_type_): 
            z_bins (list): a list of values utilizing a height parameter to segment the point clouds of occupied and free
            resolution_meter2pixel (_type_): 
            free_index (_type_): representative values of navigable areas on the map
            occupied_index (_type_): representative values of obstacle areas on the map

        Returns:
            np.array: top down map in shape (map_size, map_size)
        """        
        n_z_bins = len(z_bins)+1
        
        isnotnan = np.logical_not(np.isnan(point_clouds_2map[:,:,1]))
        # transform points meter to pixel
        X_bin = np.round(point_clouds_2map[:,:,0] / resolution_meter2pixel).astype(np.int32)
        Y_bin = np.round(point_clouds_2map[:,:,1] / resolution_meter2pixel).astype(np.int32)
        """     function explaination 
                np.digitize : split the point according to the z_bins
                example:
                    z_bins = [1] ; points that lower than 1 is 0 else 1 
        """
        Z_bin = np.digitize(point_clouds_2map[:,:,2], bins=z_bins).astype(np.int32)
        
        # filter out the points outside the map and nan
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin ) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        indr = ind.ravel()
        isvalidr = isvalid.ravel().astype(np.int32)
        count = np.bincount(indr, isvalidr, minlength=map_size*map_size*n_z_bins)
        count = count[:map_size*map_size*n_z_bins]
        count = np.reshape(count, [map_size, map_size, n_z_bins])
             
        map = np.zeros((count.shape[0],count.shape[1]))
        free_mask = count[:,:,0] > 0
        map[free_mask] = free_index
        occupied_mask = count[:,:,1] > 0
        map[occupied_mask] = occupied_index

        return map 
    
    def transformation_quatrtnion2heading(self, rotation:quaternion):
        quat = quaternion_to_list(rotation)
        q = R.from_quat(quat)
        heading = q.as_rotvec()[1]
        return heading
    
    def transformation_pointcloud2semanticmap(self, point_clouds_2map:np.array, map_size, z_bins:list, resolution_meter2pixel, free_index, semantic_obs) -> np.array :
        """project the point cloud relative to the map coordinate system to the top view

        Args:
            point_clouds_2map (np.array): 
            map_size (_type_): 
            z_bins (list): a list of values utilizing a height parameter to segment the point clouds of occupied and free
            resolution_meter2pixel (_type_): 
            free_index (_type_): representative values of navigable areas on the map
            semantic_obs (_type_): representative values of obstacle areas on the map, the shape is in (depyh_img.shape[0], depyh_img.shape[1])

        Returns:
            np.array: top down map in shape (map_size, map_size)
        """        
        n_z_bins = len(z_bins)+1
        
        isnotnan = np.logical_not(np.isnan(point_clouds_2map[:,:,1]))
        # transform points meter to pixel
        X_bin = np.round(point_clouds_2map[:,:,0] / resolution_meter2pixel).astype(np.int32)
        Y_bin = np.round(point_clouds_2map[:,:,1] / resolution_meter2pixel).astype(np.int32)
        """     function explaination 
                np.digitize : split the point according to the z_bins
                example:
                    z_bins = [1] ; points that lower than 1 is 0 else 1 
        """
        Z_bin = np.digitize(point_clouds_2map[:,:,2], bins=z_bins).astype(np.int32)
        
        # filter out the points outside the map and nan
        isvalid = np.array([X_bin >= 0, X_bin < map_size, Y_bin >= 0, Y_bin < map_size,
                            Z_bin >= 0, Z_bin < n_z_bins, isnotnan])
        isvalid = np.all(isvalid, axis=0)

        ind = (Y_bin * map_size + X_bin ) * n_z_bins + Z_bin
        ind[np.logical_not(isvalid)] = 0
        indr = ind.ravel()
        isvalidr = isvalid.ravel().astype(np.int32)
        count = np.bincount(indr, isvalidr, minlength=map_size*map_size*n_z_bins)
        count = count[:map_size*map_size*n_z_bins]
        count = np.reshape(count, [map_size, map_size, n_z_bins])
             
        map = np.zeros((count.shape[0],count.shape[1]))
        free_mask = count[:,:,0] > 0
        occupied_mask = count[:,:,1] > 0
        for y in range(X_bin.shape[0]):
            for x in range(X_bin.shape[1]):
                if Y_bin[y,x] >= 0 and \
                Y_bin[y,x] < map_size and \
                X_bin[y,x] >= 0 and \
                X_bin[y,x] < map_size:
                    if occupied_mask[Y_bin[y,x],X_bin[y,x]]:
                        map[Y_bin[y,x],X_bin[y,x]] = semantic_obs[y,x]  
                    elif free_mask[Y_bin[y,x],X_bin[y,x]]:
                        map[Y_bin[y,x],X_bin[y,x]] = free_index

        return map 
    
    
class Mode_Selector:
    
    def __init__(self) -> None:
        pass
    
            
class Action_Space:
    move_forward = 1
    turn_left = 2
    turn_right = 3
    
    
class Application(Geometry_Tools):
    
    def __init__(self, image_resolution, fov, depth_threshold, resolution_meter2pixel, map_size, camera_height, free_index, occupied_index) -> None:
        super().__init__(image_resolution, fov, camera_height)
        self._resolution_meter2pixel = resolution_meter2pixel
        self._depth_threshold = depth_threshold
        self._map_size = map_size
        self.pos2map = Pos2Map(self._map_size/2+1, self._map_size/2+1, 0)
        self.pos2world = Pos2World(None, None, None, None)
        self._free_index = free_index
        self._occupied_index = occupied_index
    
    def parse_semantic_pointclouds(self, depth_img:np.array, semantic_obs:np.array, semantic_anno):
        """Parse the point cloud dictionary with semantic annotation and the average coordinate dictionary of each 
            semantically annotated object in the robot camera coordinate system

        Args:
            depth_img (np.array): In shape (width, depth, 1)
            semantic_obs (np.array): In shape (width, depth)
            semantic_anno (_type_): _description_

        Returns:
            mapping_semantic: dictionary of all points corresponding to each label in the semantic_obs 
            occupied_pc: dictionary of average points corresponding to each label in the semantic_obs 
        """        
        # filter out points that exceed a certain distance 
        depth_img[depth_img > self._depth_threshold] = np.NaN
        # parse point clouds relative to the robot camera coordinate system
        point_clouds_2robotcamera = self.transformation_camera2robotcamera(depth_img)
        # label each pixel in semantic_obs
        ## TODO：解决相同物体不同index但是同一个label的问题
        mapping_semantic = {}
        for row in range(semantic_obs.shape[0]):
            for col in range(semantic_obs.shape[1]):
                label = semantic_anno[semantic_obs[row,col]]
                if not label in mapping_semantic.keys():
                    mapping_semantic[label] = [point_clouds_2robotcamera[row,col]]
                else:
                    mapping_semantic[label].append(point_clouds_2robotcamera[row,col])
        # remove the label that less than 50 pixels and unusual label
        occupied_pc = {}
        for k,v in mapping_semantic.items():
            if len(v) < 50:
                continue
            elif k in ['floor','ceiling','misc','wall','objects','void']:
                continue
            else:
                occupied_pc[k] = (sum(v)/len(v)).tolist()
                
        return mapping_semantic, occupied_pc
    
    def parse_depth_topdownmap(self, depth_img:np.array) -> np.array:
        """project depth image into the top down map

        Args:
            depth_img (np.array): in shape (width, height, 1)

        Returns:
            np.array: map in shape (map_size, map_size) which value 0 stands for unknow space, 
                        self._free_index stands for free space, self._occupied_index stands for occupied space
        """        
        # filter out points that exceed a certain distance 
        depth_img[depth_img > self._depth_threshold] = np.NaN
        # parse point clouds relative to the robot camera coordinate system
        point_clouds_2robotcamera = self.transformation_camera2robotcamera(depth_img)
        # parse point clouds relative to the robot base coordinate system
        point_clouds_2robotbase = self.transformation_robotcamera2base(point_clouds_2robotcamera)
        # parse point clouds relative to the map coordinate system
        point_clouds_2map = self.transformation_robotbase2map(point_clouds_2robotbase, self.pos2map, self._resolution_meter2pixel)
        # project the point clouds relative to the map coordinate system to top down map
        occupied_map = self.transformation_pointcloud2occupiedmap(point_clouds_2map, self._map_size, [self._camera_height], self._resolution_meter2pixel, self._free_index, self._occupied_index)
        return occupied_map
        
    def parse_semantic_topdownmap(self, depth_img:np.array, semantic_img:np.array) -> np.array:
        # filter out points that exceed a certain distance 
        depth_img[depth_img > self._depth_threshold] = np.NaN
        # parse point clouds relative to the robot camera coordinate system
        point_clouds_2robotcamera = self.transformation_camera2robotcamera(depth_img)
        # parse point clouds relative to the robot base coordinate system
        point_clouds_2robotbase = self.transformation_robotcamera2base(point_clouds_2robotcamera)
        # parse point clouds relative to the map coordinate system
        point_clouds_2map = self.transformation_robotbase2map(point_clouds_2robotbase, self.pos2map, self._resolution_meter2pixel)
        # project the point clouds relative to the map coordinate system to top down map
        semantic_map = self.transformation_pointcloud2semanticmap(point_clouds_2map, self._map_size, [self._camera_height], self._resolution_meter2pixel, self._free_index, semantic_img)
        return semantic_map
    
    def update_pos2map_by_action(self, forward_step2tenmeter, turn_angle2degree, action) -> None:
        if action == Action_Space.move_forward:
            self.pos2map.x = self.pos2map.x + forward_step2tenmeter*np.cos(self.pos2map.heading)/self._resolution_meter2pixel
            self.pos2map.y = self.pos2map.y + forward_step2tenmeter*np.sin(self.pos2map.heading)/self._resolution_meter2pixel
        elif action == Action_Space.turn_left:
            self.pos2map.heading = self.pos2map.heading + turn_angle2degree*np.pi/180. 
        elif action == Action_Space.turn_right:
            self.pos2map.heading = self.pos2map.heading - turn_angle2degree*np.pi/180. 
  
        if self.pos2map.heading > np.pi*2:
            self.pos2map.heading -= np.pi*2
        elif self.pos2map.heading < 0:
            self.pos2map.heading += np.pi*2
    
    def update_pos2map_by_cooardinate(self, tgt_pos2world:list=None, tgt_rot2world:quaternion=None) -> None:
        """_summary_

        Args:
            tgt_pos2world (list, optional): _description_. Defaults to None.
            tgt_rot2world (quaternion)
            tgt_heading2world (_type_, optional): in radius. Defaults to None.
        """        
        if not tgt_rot2world is None:
            tgt_heading2world = self.transformation_quatrtnion2heading(tgt_rot2world)
            if tgt_heading2world > np.pi*2:
                tgt_heading2world -= np.pi*2
            elif tgt_heading2world < 0:
                tgt_heading2world += np.pi*2
            
        if self.pos2world.x is None:
            self.pos2world.x = tgt_pos2world[0]
            self.pos2world.y = tgt_pos2world[1]
            self.pos2world.z = tgt_pos2world[2]
            self.pos2world.heading = tgt_heading2world
        else:
            if not tgt_pos2world is None and not (abs(tgt_pos2world[0]-self.pos2world.x)+abs(tgt_pos2world[2]-self.pos2world.z)<0.001):
                xt, yt, zt = tgt_pos2world
                delta_heading2world = np.arctan((xt-self.pos2world.x)/(zt-self.pos2world.z))
                delta_heading2world = delta_heading2world if (self.pos2world.heading<np.pi/2 or self.pos2world.heading>np.pi*3/2) else delta_heading2world+np.pi
                delta_distance2map = np.linalg.norm([(xt-self.pos2world.x)/10, (zt-self.pos2world.z)/10]) / self._resolution_meter2pixel 
                delta_heading2curheading = delta_heading2world - self.pos2world.heading
                delta_heading2map = delta_heading2curheading + self.pos2map.heading
                self.pos2map.x = self.pos2map.x + np.cos(delta_heading2map) * delta_distance2map
                self.pos2map.y = self.pos2map.y + np.sin(delta_heading2map) * delta_distance2map
                self.pos2world.x = xt
                self.pos2world.y = yt
                self.pos2world.z = zt
                

            if not tgt_heading2world is None:
                delta_heading2world = tgt_heading2world - self.pos2world.heading
                self.pos2world.heading = tgt_heading2world
                if self.pos2world.heading > np.pi*2:
                    self.pos2world.heading -= np.pi*2
                elif self.pos2world.heading < 0:
                    self.pos2world.heading += np.pi*2
                self.pos2map.heading += delta_heading2world
                if self.pos2map.heading > np.pi*2:
                    self.pos2map.heading -= np.pi*2
                elif self.pos2map.heading < 0:
                    self.pos2map.heading += np.pi*2

    def update_occupied_map(self, new_occupied_map, old_occupied_map):
        mask_free_reigon = new_occupied_map == self._free_index
        old_occupied_map[mask_free_reigon] = self._free_index
        mask_occupied_reigon = new_occupied_map == self._occupied_index
        old_occupied_map[mask_occupied_reigon] = self._occupied_index
        return old_occupied_map
    
    def update_semantic_map(self, new_semantic_map, old_semantic_map):
        mask = new_semantic_map > 0
        for y in range(old_semantic_map.shape[0]):
            for x in range(old_semantic_map.shape[1]):
                if mask[y,x]:
                    old_semantic_map[y,x] = new_semantic_map[y,x] 
        return old_semantic_map
        

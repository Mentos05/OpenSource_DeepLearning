import cv2
import base64
import numpy as np
from matplotlib.cm import get_cmap
import matplotlib
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize
from scipy.spatial import cKDTree
import itertools

class SocialDistancing():
    def __init__(self, image_shape=(1080,1920,3), map_view=False, homography_matrix=False, max_distance_detection=80, max_distance_detection_crowd=40, min_crowd_size=3):
        self.map_view = map_view
        self.homography_matrix = homography_matrix
        self.max_distance_detection = max_distance_detection
        self.max_distance_detection_crowd = max_distance_detection_crowd
        self.min_crowd_size = min_crowd_size
        if map_view == True:
            P = np.array([[0,image_shape[1],image_shape[1],0],[0,0,image_shape[0],image_shape[0]],[1,1,1,1]])
            h__ = homography_matrix.dot(P)
            self.max_x, self.max_y = int(np.max(h__[0] / h__[2])), int(np.max(h__[1] / h__[2]))
    
    # Function which is called by SAS Event Stream Processing - Output: Combined image of camera view and map view
    def func_social_distancing(self, image, detections):
        func_image = image.copy()
        # Get objects from model scoring
        objs = []
        for detection in detections:
            if detection[0] == b'person':
                x, y, w, h = detection[2][0]*(1920/608), detection[2][1]*(1080/608), detection[2][2]*(1920/608), detection[2][3]*(1080/608)
                # only accept small objects / remove errors
                if w*h > 25000:
                    continue
                x1, x2, y1, y2 = int(x-w/2), int(x+w/2), int(y-h/2), int(y+h/2)
                objs.append([x,y,x1,y1,x2,y2])
                #print('objectsize:{}, höhe={},breite={}'.format(h*w, h, w))
        objs = np.array(objs, dtype=np.int32)
        # Calculate distances and visualize objects on image and map
        func_image, map2d = self.calc_visualize(func_image,objs)
        if self.map_view == True:
            # Create combined image (for return if no objects found)
            offset = 20
            combined = np.zeros([1080,1920+self.max_x+offset,3], np.uint8)
            # Write number of persons on image, colored based on number of persons
            cmap = LinearSegmentedColormap.from_list("", ["green","yellow","red"])
            norm = matplotlib.colors.Normalize(vmin=25, vmax=35, clip=True)
            color = np.array(cmap(norm(len(objs)))[0:3])*255
            color = (color[2],color[1],color[0])
            cv2.putText(map2d, 'Number of persons:', (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.putText(map2d, str('{}'.format(len(objs))), (400,50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            # Combine image and map to single output-image
            combined[:1080,:1920] = func_image
            combined[:self.max_y,1920+offset:1920+offset+self.max_x] = map2d
            # Encode image for SAS Event Stream Processing
            scored_image = combined
        else:
            scored_image = func_image
        scored_image = cv2.cvtColor(scored_image, cv2.COLOR_BGR2RGB)
        return scored_image
    
    # Helpfer function to return transformed x,y-values given a homography matrix
    def toworld(self,xy):
        imagepoint = [xy[0], xy[1], 1]
        worldpoint = np.array(np.dot(np.array(self.homography_matrix),imagepoint))
        scalar = worldpoint[2]
        xworld = int(worldpoint[0]/scalar)
        yworld = int(worldpoint[1]/scalar)
        return xworld, yworld

    # Helper function to return x,y-coordinate pairs and distance for drawing lines
    def get_line_coordinates(self, pair,objects):
        pair_dst = pair[2]
        pair_coords = pair[0:2]
        if self.map_view == True:
            pair_coords = objects[[pair_coords[0],pair_coords[1]]][:,[0,1,6,7]].flatten()
        else:
            pair_coords = objects[[pair_coords[0],pair_coords[1]]][:,[0,1]].flatten()
        pair_data = np.append(pair_coords,pair_dst)
        return pair_data

    # Crowd detection function
    def crowd_detection(self,objects, max_distance, viz_type, min_persons):
        if self.map_view == True:
            tree_ball = cKDTree(objects[:,[6,7]])
        else:
            tree_ball = cKDTree(objects[:,[0,1]])
        crowds = tree_ball.query_ball_tree(tree_ball, max_distance)
        crowds.sort()
        crowds = list(object for object,_ in itertools.groupby(crowds))
        crowd_size = np.array([len(crowd) for crowd in crowds])
        crowd_coords = np.array([], dtype=np.int16)
        if viz_type == 'image':
            for crowd in crowds:
                coords = self.get_crowd_coordinates(crowd, objects, 10, 'image')
                crowd_coords = np.append(crowd_coords, coords)
        if viz_type == 'map':
            for crowd in crowds:
                coords = self.get_crowd_coordinates(crowd, objects, 10, 'map')
                crowd_coords = np.append(crowd_coords, coords)
        crowd_coords = crowd_coords.reshape(int(len(crowd_coords)/4),4)
        crowd_data = np.column_stack((crowd_coords,crowd_size))
        crowd_data = crowd_data[np.argwhere(crowd_data[:,4] >= min_persons)].reshape(len(np.argwhere(crowd_data[:,4] >= min_persons)),5)
        return crowd_data

    # Helper function to retrieve crowd coordinates
    def get_crowd_coordinates(self,crowd, objects, offset, viz_type):
        if viz_type == 'image':
            min_x, min_y = np.min(objects[crowd,2]), np.min(objects[crowd,3])
            max_x, max_y = np.max(objects[crowd,4]), np.max(objects[crowd,5])
        if viz_type == 'map':
            min_x, min_y = np.min(objects[crowd,6]), np.min(objects[crowd,7])
            max_x, max_y = np.max(objects[crowd,6]), np.max(objects[crowd,7])
        return np.array([min_x-offset, min_y-offset, max_x+offset, max_y+offset], dtype=np.int16)

    # Helper function to check if one box is inside another
    def intersection(self,box0,box1):
        if box0[0] >= box1[0] and box0[1] >= box1[1]: 
            if box0[2] <= box1[2] and box0[3] <= box1[3]:
                return True #box inside
        return False #box not inside

    # Helper function to suppress detected crowds that are inside another crowd (only show main crowd)
    def crowd_suppression(self,crowd_data):
        crowd_data_filtered = np.array([],dtype=np.int16)
        for i in range(len(crowd_data)):
            is_inside = False
            for j in range(len(crowd_data)):
                if i==j:
                    continue
                if self.intersection(crowd_data[i],crowd_data[j]):
                    is_inside = True
            if is_inside == False:
                crowd_data_filtered = np.append(crowd_data_filtered, crowd_data[i])
        crowd_data_filtered = crowd_data_filtered.reshape(int(len(crowd_data_filtered)/5),5)
        x = np.random.rand(crowd_data_filtered.shape[1])
        y = crowd_data_filtered.dot(x)
        unique, index = np.unique(y, return_index = True)
        crowd_data_filtered = crowd_data_filtered[index]
        return crowd_data_filtered

    # Main function to calculate distances and to create visualization
    def calc_visualize(self,func_image, objects):
        if self.map_view == True:
            if len(objects) == 0:
                map2d = np.ones([self.max_y, self.max_x,3],dtype=np.int8)
                return func_image, map2d
            # Create Colormap
            cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
            # Create empty Map and combined image
            map2d = np.ones([self.max_y, self.max_x,3],dtype=np.int8)
            # Transform x,y coordinates (adapt to camera angle based on calculated homography)
            objects_x_y_transformed = np.apply_along_axis(self.toworld, 1, objects[:,0:2])
            objects = np.column_stack((objects, objects_x_y_transformed)) #x,y,x1,y1,x2,y2,map_x,map_y
            # Get distances for transformed coordinates for all objects using KD-Tree - set distance to 255 if distance > threshold
            tree = cKDTree(objects_x_y_transformed)
            t_dst = tree.sparse_distance_matrix(tree, self.max_distance_detection)
            t_dst = t_dst.toarray()
            t_dst = np.array(t_dst, dtype=np.int32)
            t_dst2 = t_dst.copy()
            t_dst2[np.where(t_dst2==0)]=255
            objects = np.column_stack((objects,np.min(t_dst2,1))) #x,y,x1,y1,x2,y2,map_x,map_y,distance -> get minimum distance to another object for each object (to draw bounding boxes and points)
            # Create distance lines
            near_pairs = np.column_stack((np.argwhere(t_dst > 0),t_dst[np.nonzero(t_dst)]))
            # Get coordinates for drawing lines
            if len(near_pairs) > 0:
                near_pairs = np.apply_along_axis(self.get_line_coordinates, 1, near_pairs, objects)
            # Draw object bounding boxes, colored based on minimum distance to another person
            for object_ in objects:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=self.max_distance_detection, clip=True)
                color = np.array(cmap(norm(object_[8]))[0:3])*255
                color = (color[2],color[1],color[0])
                if int(object_[8]) < 255:
                    cv2.putText(func_image, str(int(object_[8])), (object_[0],object_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.rectangle(func_image, (int(object_[2]),int(object_[3])), (int(object_[4]),int(object_[5])), color, 2) #x1,y1,x2,y2,color,linestrength 
                cv2.circle(map2d, (int(object_[6]),int(object_[7])), 10, color, -1)
            # Draw lines between objects, colored based on distance
            for line_ in near_pairs:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=self.max_distance_detection, clip=True)
                color = np.array(cmap(norm(line_[8]))[0:3])*255
                color = (color[2],color[1],color[0])
                text_pt_x = int((int(line_[0])+int(line_[4])) / 2)
                text_pt_y = int((int(line_[1])+int(line_[5])) / 2)
                cv2.putText(func_image, str(int(line_[8])), (text_pt_x,text_pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.line(func_image,(int(line_[0]),int(line_[1])),(int(line_[4]),int(line_[5])),color,2)
                text_pt_x_map = int((int(line_[2])+int(line_[6])) / 2)
                text_pt_y_map = int((int(line_[3])+int(line_[7])) / 2)
                cv2.putText(map2d, str(int(line_[8])), (text_pt_x_map,text_pt_y_map), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.line(map2d,(int(line_[2]),int(line_[3])),(int(line_[6]),int(line_[7])),color,2)
            # Detect and draw crowds for image (based on transformed coordinates)
            crowd_data = self.crowd_detection(objects, self.max_distance_detection_crowd, 'image', self.min_crowd_size)
            crowd_data = self.crowd_suppression(crowd_data)
            for crowd in crowd_data:
                border_offset=3
                (label_width, label_height), baseline = cv2.getTextSize('Crowdsize: X', cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(func_image,(crowd[0],crowd[1]),(crowd[0]+label_width+10,crowd[1]-label_height-border_offset-10),(255,0,0),-1)
                cv2.putText(func_image, 'Crowdsize: {}'.format(crowd[4]), (crowd[0]+5, crowd[1]-border_offset-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(func_image, (int(crowd[0]),int(crowd[1])), (int(crowd[2]),int(crowd[3])), (255,0,0), 2)
            # Detect and draw crowds for map (based on transformed coordinates)
            crowd_data = self.crowd_detection(objects, self.max_distance_detection_crowd, 'map', self.min_crowd_size)
            crowd_data = self.crowd_suppression(crowd_data)
            for crowd in crowd_data:
                border_offset=3
                (label_width, label_height), baseline = cv2.getTextSize('Crowdsize: X', cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(map2d,(crowd[0],crowd[1]),(crowd[0]+label_width+10,crowd[1]-label_height-border_offset-10),(255,0,0),-1)
                cv2.putText(map2d, 'Crowdsize: {}'.format(crowd[4]), (crowd[0]+5, crowd[1]-border_offset-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(map2d, (int(crowd[0]),int(crowd[1])), (int(crowd[2]),int(crowd[3])), (255,255,255), 2)
            ### If no homography available ...
        if self.map_view == False:
            map2d = None
            # Create Colormap
            cmap = LinearSegmentedColormap.from_list("", ["red","yellow","green"])
            # Get distances for coordinates for all objects using KD-Tree - set distance to 255 if distance > threshold
            objects_x_y =  objects[:,0:2]#np.apply_along_axis(toworld, 1, objects[:,0:2])
            tree = cKDTree(objects_x_y)
            t_dst = tree.sparse_distance_matrix(tree, self.max_distance_detection)
            t_dst = t_dst.toarray()
            t_dst = np.array(t_dst, dtype=np.int32)
            t_dst2 = t_dst.copy()
            t_dst2[np.where(t_dst2==0)]=255
            objects = np.column_stack((objects,np.min(t_dst2,1))) #x,y,x1,y1,x2,y2,map_x,map_y,distance -> get minimum distance to another object for each object (to draw bounding boxes and points)
            # Create distance lines
            near_pairs = np.column_stack((np.argwhere(t_dst > 0),t_dst[np.nonzero(t_dst)]))
            # Get coordinates for drawing lines
            if len(near_pairs) > 0:
                near_pairs = np.apply_along_axis(self.get_line_coordinates, 1, near_pairs, objects)
            # Draw object bounding boxes, colored based on minimum distance to another person
            for object_ in objects:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=self.max_distance_detection, clip=True)
                color = np.array(cmap(norm(object_[6]))[0:3])*255
                color = (color[2],color[1],color[0])
                if int(object_[6]) < 255:
                    cv2.putText(func_image, str(int(object_[6])), (object_[0],object_[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.rectangle(func_image, (int(object_[2]),int(object_[3])), (int(object_[4]),int(object_[5])), color, 2) #x1,y1,x2,y2,color,linestrength 
            # Draw lines between objects, colored based on distance
            for line_ in near_pairs:
                norm = matplotlib.colors.Normalize(vmin=0, vmax=self.max_distance_detection, clip=True)
                color = np.array(cmap(norm(line_[4]))[0:3])*255
                color = (color[2],color[1],color[0])
                text_pt_x = int((int(line_[0])+int(line_[2])) / 2)
                text_pt_y = int((int(line_[1])+int(line_[3])) / 2)
                cv2.putText(func_image, str(int(line_[3])), (text_pt_x,text_pt_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)
                cv2.line(func_image,(int(line_[0]),int(line_[1])),(int(line_[2]),int(line_[3])),color,2)
            # Detect and draw crowds for image
            crowd_data = self.crowd_detection(objects, self.max_distance_detection_crowd, 'image', min_crowd_size)
            crowd_data = self.crowd_suppression(crowd_data)
            for crowd in crowd_data:
                border_offset=3
                (label_width, label_height), baseline = cv2.getTextSize('Crowdsize: X', cv2.FONT_HERSHEY_DUPLEX, 0.6, 1)
                cv2.rectangle(func_image,(crowd[0],crowd[1]),(crowd[0]+label_width+10,crowd[1]-label_height-border_offset-10),(255,0,0),-1)
                cv2.putText(func_image, 'Crowdsize: {}'.format(crowd[4]), (crowd[0]+5, crowd[1]-border_offset-5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(func_image, (int(crowd[0]),int(crowd[1])), (int(crowd[2]),int(crowd[3])), (255,0,0), 2)
        return func_image, map2d
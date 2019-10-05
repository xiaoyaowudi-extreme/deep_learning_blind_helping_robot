import numpy as np
import tensorflow as tf
from collections import defaultdict
from io import StringIO
from PIL import Image

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util


class recognition(object):
    def __init__(self, pre_defined_height, pre_defined_width, graph_path, device_location, use_gpu = False, fast_recognition = True, fast_recognition_image = None):
        self.rows = pre_defined_height
        self.cols = pre_defined_width
        if use_gpu:
            self.device_name = '/gpu:' + str( int ( device_location ) )
        else:
            self.device_name = '/cpu:' + str( int ( device_location ) )
        self.graph = tf.Graph()
        with self.graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile( graph_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString( serialized_graph )
                tf.import_graph_def( od_graph_def, name='')
        with self.graph.as_default():
            self.session = tf.Session()
        with self.graph.as_default():
            with tf.device( self.device_name ):
                #Get handles to input and output tensors
                ops = tf.get_default_graph().get_operations()
                all_tensor_names = {output.name for op in ops for output in op.outputs}
                #print(all_tensor_names)
                self.tensor_dict = {}
                for key in [
                        'num_detections', 'detection_boxes', 'detection_scores',
                        'detection_classes', 'detection_masks'
                ]:
                    tensor_name = key + ':0'
                    if tensor_name in all_tensor_names:
                        self.tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                                tensor_name)
                if 'detection_masks' in self.tensor_dict:
                    # The following processing is only for single image
                    detection_boxes = tf.squeeze(self.tensor_dict['detection_boxes'], [0])
                    detection_masks = tf.squeeze(self.tensor_dict['detection_masks'], [0])
                    # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                    real_num_detection = tf.cast(self.tensor_dict['num_detections'][0], tf.int32)
                    detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                    detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                    detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                            detection_masks, detection_boxes, self.rows, self.cols)
                    detection_masks_reframed = tf.cast(
                            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
                    # Follow the convention by adding back the batch dimension
                    self.tensor_dict['detection_masks'] = tf.expand_dims(
                            detection_masks_reframed, 0)
                self.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
                if fast_recognition == True:
                    _ = self.session.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(fast_recognition_image, 0)})
    def load_image_into_numpy_array(image):
        (im_width, im_height) = image.size
        return np.array( image.getdata() ).reshape( (im_height, im_width, 3) ).astype( np.uint8 )
    def recognize(self, image):
        with self.graph.as_default():
            with tf.device( self.device_name ):
                output_dict = self.session.run(self.tensor_dict, feed_dict={self.image_tensor: np.expand_dims(image, 0)})
                output_dict['num_detections'] = int(output_dict['num_detections'][0])
                output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.uint8)
                output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                output_dict['detection_scores'] = output_dict['detection_scores'][0]
                if 'detection_masks' in output_dict:
                    output_dict['detection_masks'] = output_dict['detection_masks'][0]
        return output_dict
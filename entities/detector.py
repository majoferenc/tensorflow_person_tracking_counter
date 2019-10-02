import numpy as np
import tensorflow as tf
import os
cwd = os.path.dirname(os.path.realpath(__file__))
import utils.visualization_utils as visualization_utils

class PersonDetector(object):
    def __init__(self):
        self.cap = None
        self.cv2 = None
        self.roi_increment = None
        self.roi_decrement = None
        self.deviation_increment = 1  # the constant that represents the object counting increment
        self.deviation_decrement = 1
        self.total_passed_people = 0
        self.person_boxes = []

        os.chdir(cwd)

        PATH_TO_CKPT = '../ssd_mobilenet_v2/frozen_inference_graph.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.compat.v1.GraphDef()
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            self.sess = tf.compat.v1.Session(graph=self.detection_graph)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            self.scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')

    # Helper function to convert image into numpy array
    def load_image_into_numpy_array(self, image):
        (im_width, im_height) = image.size
        return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)
        # Helper function to convert normalized box coordinates to pixels

    def box_normal_to_pixel(self, box, dim):

        height, width = dim[0], dim[1]
        box_pixel = [int(box[0] * height), int(box[1] * width), int(box[2] * height), int(box[3] * width)]
        return np.array(box_pixel)

    def get_localization(self, image):
        category_index = {1: {'id': 1, 'name': u'person'}}

        with self.detection_graph.as_default():
            image_expanded = np.expand_dims(image, axis=0)
            (boxes, scores, classes, num_detections) = self.sess.run(
                [self.boxes, self.scores, self.classes, self.num_detections],
                feed_dict={self.image_tensor: image_expanded})

            # Visualization of the results of a detection.
            counter, csv_line, counting_mode = visualization_utils.visualize_boxes_and_labels_on_image_array_x_axis(
                self.cap.get(1),
                image,
                1,
                0,
                np.squeeze(
                    boxes),
                np.squeeze(
                    classes).astype(
                    np.int32),
                np.squeeze(
                    scores),
                category_index,
                x_reference=self.roi_increment,
                deviation=self.deviation_increment,
                use_normalized_coordinates=True,
                line_thickness=4)

            # Visualization of the results of a detection.
            counter2, csv_line2, counting_mode2 = visualization_utils.visualize_boxes_and_labels_on_image_array_x_axis(
                self.cap.get(1),
                image,
                1,
                0,  # color recognition
                np.squeeze(
                    boxes),
                np.squeeze(
                    classes).astype(
                    np.int32),
                np.squeeze(
                    scores),
                category_index,
                x_reference=self.roi_decrement,
                deviation=self.deviation_decrement,
                use_normalized_coordinates=True,
                line_thickness=4)

            if counter == 1:
                self.cv2.line(image, (self.roi_increment, 0), (self.roi_increment, 720), (0, 0xFF, 0), 5)
            else:
                self.cv2.line(image, (self.roi_increment, 0), (self.roi_increment, 720), (0, 0, 0xFF), 5)

            self.total_passed_people = self.total_passed_people + counter

            if counter2 == 1:
                self.cv2.line(image, (self.roi_decrement, 0), (self.roi_decrement, 720), (0, 0xFF, 0), 5)
            else:
                self.cv2.line(image, (self.roi_decrement, 0), (self.roi_decrement, 720), (0xFF, 0, 0), 5)

            total_passed_people = self.total_passed_people - counter2

            font = self.cv2.FONT_HERSHEY_SIMPLEX
            self.cv2.putText(
                image,
                'Detected People: ' + str(total_passed_people),
                (10, 45),
                font,
                0.8,
                (0, 0xFF, 0xFF),
                2,
                self.cv2.FONT_HERSHEY_SIMPLEX,
            )

            self.cv2.putText(
                image,
                'Increment',
                (545, self.roi_increment - 10),
                font,
                0.6,
                (0, 0, 0xFF),
                2,
                self.cv2.LINE_AA,
            )

            self.cv2.putText(
                image,
                'Decrement',
                (525, self.roi_decrement - 10),
                font,
                0.6,
                (0xFF, 0, 0),
                2,
                self.cv2.LINE_AA,
            )
            boxes = np.squeeze(boxes)
            classes = np.squeeze(classes)
            scores = np.squeeze(scores)

            cls = classes.tolist()

            idx_vec = [i for i, v in enumerate(cls) if ((v == 1) and (scores[i] > 0.3))]

            if len(idx_vec) == 0:
                pass
            else:
                tmp_person_boxes = []
                for idx in idx_vec:
                    dim = image.shape[0:2]
                    box = self.box_normal_to_pixel(boxes[idx], dim)
                    box_h = box[2] - box[0]
                    box_w = box[3] - box[1]
                    ratio = box_h / (box_w + 0.01)

                    tmp_person_boxes.append(box)
                    print(box, ', confidence: ', scores[idx], 'ratio:', ratio)

                self.person_boxes = tmp_person_boxes

        return self.person_boxes
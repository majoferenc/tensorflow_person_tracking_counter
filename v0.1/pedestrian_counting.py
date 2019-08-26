from utils import backbone
import tensorflow as tf
import cv2
import numpy as np
from utils import visualization_utils as vis_util

total_passed_people = 0  # using it to count vehicles
font = cv2.FONT_HERSHEY_SIMPLEX
pedestrians = []


def object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled,
                                      roi_increment, roi_decrement, deviation_increment, deviation_decrement):
    total_passed_people = 0

    # input video
    cap = cv2.VideoCapture(input_video)

    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    total_passed_people = 0
    speed = "waiting..."
    direction = "waiting..."
    size = "waiting..."
    color = "waiting..."
    counting_mode = "..."
    width_heigh_taken = True
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # for all the frames that are extracted from input video
            while (cap.isOpened()):
                ret, frame = cap.read()

                if not ret:
                    print("end of the video file...")
                    break

                input_frame = frame

                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(input_frame, axis=0)

                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                    [detection_boxes, detection_scores, detection_classes, num_detections],
                    feed_dict={image_tensor: image_np_expanded})

                # Visualization of the results of a detection.
                counter, csv_line, counting_mode = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(cap.get(1),
                                                                                                             input_frame,
                                                                                                             1,
                                                                                                             is_color_recognition_enabled,
                                                                                                             np.squeeze(
                                                                                                                 boxes),
                                                                                                             np.squeeze(
                                                                                                                 classes).astype(
                                                                                                                 np.int32),
                                                                                                             np.squeeze(
                                                                                                                 scores),
                                                                                                             category_index,
                                                                                                             x_reference=roi_increment,
                                                                                                             deviation=deviation_increment,
                                                                                                             use_normalized_coordinates=True,
                                                                                                             line_thickness=4)

                print(boxes)
                print(counter)
                print(counting_mode)

                # Visualization of the results of a detection.
                counter2, csv_line2, counting_mode2 = vis_util.visualize_boxes_and_labels_on_image_array_x_axis(
                    cap.get(1),
                    input_frame,
                    1,
                    is_color_recognition_enabled,
                    np.squeeze(
                        boxes),
                    np.squeeze(
                        classes).astype(
                        np.int32),
                    np.squeeze(
                        scores),
                    category_index,
                    x_reference=roi_decrement,
                    deviation=deviation_decrement,
                    use_normalized_coordinates=True,
                    line_thickness=4)

                if counter == 1:
                    cv2.line(input_frame, (roi_increment, 0), (roi_increment, height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (roi_increment, 0), (roi_increment, height), (0, 0, 0xFF), 5)

                total_passed_people = total_passed_people + counter

                if counter2 == 1:
                    cv2.line(input_frame, (roi_decrement, 0), (roi_decrement, height), (0, 0xFF, 0), 5)
                else:
                    cv2.line(input_frame, (roi_decrement, 0), (roi_decrement, height), (0xFF, 0, 0), 5)

                total_passed_people = total_passed_people - counter2


                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(
                    input_frame,
                    'Detected Pedestrians: ' + str(total_passed_people),
                    (10, 45),
                    font,
                    0.8,
                    (0, 0xFF, 0xFF),
                    2,
                    cv2.FONT_HERSHEY_SIMPLEX,
                )

                cv2.putText(
                    input_frame,
                    'Increment',
                    (545, roi_increment - 10),
                    font,
                    0.6,
                    (0, 0, 0xFF),
                    2,
                    cv2.LINE_AA,
                )

                cv2.putText(
                    input_frame,
                    'Decrement',
                    (525, roi_decrement - 10),
                    font,
                    0.6,
                    (0xFF, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.putText(
                    input_frame,
                    'FPS: ' + fps.__str__(),
                    (10, 20),
                    font,
                    0.6,
                    (0xFF, 0, 0),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow('Live object detection', cv2.resize(input_frame, (width, height)))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()


input_video = "./input_images_and_videos/pedestrian.mp4"

detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2018_01_28', 'label_map.pbtxt')

is_color_recognition_enabled = 0 # set it to 1 for enabling the color prediction for the detected objects
roi_increment = 385 # roi line position
roi_decrement = 365
deviation_increment = 1 # the constant that represents the object counting area
deviation_decrement = 1 # the constant that represents the object counting area
object_counting(input_video, detection_graph, category_index, is_color_recognition_enabled, roi_increment, roi_decrement, deviation_increment, deviation_decrement) # counting all the objects

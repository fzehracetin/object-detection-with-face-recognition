from __future__ import print_function
import sys
import cv2
import numpy as np
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
from scipy import ndimage
from threading import Thread
import face_recognition
import pickle

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")


class Detection:
    def __init__(self, model_path, label_map_path, num_classes, f_model_path):
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.num_classes = num_classes
        self.f_model_path = f_model_path
        self.captureTuples = list()
        self.video = None
        self.outputs = {'list': [],
                        'name': []}
        self.checklistController = list()
        self.num_frame = 0
        self.checklist = list()
        self.isciler = []

        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map, max_num_classes=self.num_classes, use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        # print("category_index: ", self.category_index)

        self.detection_graph = tf.Graph()

        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(model_path, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

    def recognize_face(self, frame, distance_threshold=0.6):
        with open(self.f_model_path, 'rb') as f:
            knn_clf = pickle.load(f)

        '''cv2.imshow("cropped frame", frame)
        cv2.waitKey(0)'''
        rgb_frame = frame[:, :, ::-1]
        face_locations = face_recognition.face_locations(rgb_frame)
        if len(face_locations) == 0:
            print("yuz yok")
            return None
        print("yuz var")
        faces_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
        are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(face_locations))]

        return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
                zip(knn_clf.predict(faces_encodings), face_locations, are_matches)]

    def add_conditional_capture(self, condition, callback):
        self.captureTuples.append((condition, callback))

    def checklist_controller(self, checker):
        self.checklistController = (
            checker.on_isci_detection, checker.is_in_checklist, checker.output_reader, checker.detect_and_crop)
        self.checklist = checker.checklist

    def set_video(self, path):
        self.video = cv2.VideoCapture(path)

    def start(self):
        if self.video is None:
            raise ValueError("No video set!")
        t1 = Thread(target=self.process_video, args=())
        t1.start()
        t1.join()
        print("Video tamamlandi")
        self.checklistController[2](self.isciler)

    def dictionary_maker(self, index, detections, boxes, scores, classes, num):
        lis = []
        for i in range(int(num[0])):
            if classes[0][i] == index:
                lis.append([scores[0][i], list(boxes[0][i])])

        detections['objects'].update({self.category_index[index]['name']: lis})
        detections['meta'].update({
            self.category_index[index]['name']: len(detections['objects'][self.category_index[index]['name']])})
        del lis

    def visualize_and_write(self, frame, boxes, classes, scores, predictions, output):
        # ----------------------------------LANDSCAPE----------------------------------------
        # Draw a transparent overlay
        bottom = 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        y1 = bottom + (len(self.checklist)*30 + 60)*(len(self.isciler))
        alpha = 0.4
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (350, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        # Print user's checklists on frame
        for i in range(len(self.isciler)):
            cv2.putText(frame, str(self.isciler[i]['name']), (10, bottom), font, 1, (255, 255, 255), 2)
            bottom += 30
            checklist = list(set(self.checklist) ^ set(self.isciler[i]['list']))
            for element in checklist:
                # yesil yazar, esya var
                cv2.putText(frame, "- " + str(element), (20, bottom), font, 1, (0, 255, 0), 2)
                bottom += 30
            for j in range(len(self.isciler[i]['list'])):
                # kirmizi yazar, esya yok
                cv2.putText(frame, "- " + str(self.isciler[i]['list'][j]), (20, bottom), font, 1, (0, 0, 255), 2)
                bottom += 30
            cv2.putText(frame, "___________________", (0, bottom), font, 1, (255, 255, 255), 2)
            bottom += 60
        # -----------------------------------------------------------------------------------
        # Draw the results of the detection (aka 'visualize the results')
        vis_util.visualize_boxes_and_labels_on_image_array(
            frame,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            self.category_index,
            use_normalized_coordinates=True,
            line_thickness=8,
            min_score_thresh=0.60)  # 0.60 is default value

        # Draw the faces
        for prediction in predictions:
            for name, (top, right, bottom, left) in prediction:
                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
        '''cv2.imshow('frame', frame)
        cv2.waitKey(0)'''

        output.write(frame)

    def process_video(self):
        with tf.Session(graph=self.detection_graph) as sess:
            image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')

            output_tensors = [self.detection_graph.get_tensor_by_name('detection_boxes:0'),
                              self.detection_graph.get_tensor_by_name('detection_scores:0'),
                              self.detection_graph.get_tensor_by_name('detection_classes:0'),
                              self.detection_graph.get_tensor_by_name('num_detections:0')]

            output_path = 'output.avi'
            fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
            frames = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = 15

            frame_width = int(self.video.get(3))
            frame_height = int(self.video.get(4))
            print("frame height:", frame_height, "frame_width:", frame_width)
            # -----------------------------------PORTRAIT-----------------------------------------
            '''temp = int(frame_width / 2)
            frame_width = int(frame_height / 2)
            frame_height = temp'''
            # ------------------------------------------------------------------------------------
            output = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            frame_number = 0
            while self.video.isOpened():
                ret, frame = self.video.read()
                # cv2.imshow("frame", frame[20:25, 80:99])

                if ret is True:
                    frame_number += 1
                    # -----------------------------------PORTRAIT----------------------------------------
                    '''frame = ndimage.rotate(frame, 270)
                    frame = cv2.resize(frame, (frame_width, frame_height))'''
                    # -----------------------------------------------------------------------------------
                    (boxes, scores, classes, num) = self.process_frame(frame, sess, image_tensor, output_tensors)

                    # boxes: koordinatlar, score: yuzdelikler, classes: hangi obje oldugu int, num: index
                    detections = {'meta': {},
                                  'objects': {}}

                    # 1 -> yelek, 2 -> kask, 3 -> gozluk, 4 -> eldiven, 5 -> isci
                    # 6 -> yelek_yok, 7 -> kask_yok, 8 -> isci_yok, 9 -> eldiven_yok
                    for i in range(1, len(self.category_index) + 1):
                        self.dictionary_maker(i, detections, boxes, scores, classes, num)
                    predictions = []

                    # isci varsa yuz tanimaya git
                    if self.checklistController[0](detections) is True:
                        coords = self.checklistController[3](detections)
                        # coords[0] -> y0, coords[1] -> x0, coords[2] -> y1, coords[3] -> x1
                        for i in range(len(coords)):
                            # buldugun isci frame'ini kirp
                            cropped_frame = np.array(frame[
                                                     int(coords[i][0] * frame_height): int(coords[i][2] * frame_height),
                                                     int(coords[i][1] * frame_width): int(coords[i][3] * frame_width)
                                                     ])
                            prediction = self.recognize_face(cropped_frame)
                            # print("prediction", prediction)
                            # prediction[0][0] -> isim, prediction[0][1] -> koordinatlar
                            if prediction is not None and prediction[0][0] != 'unknown':
                                j = 0
                                flag = False
                                while j < len(self.isciler) and flag is False:
                                    if prediction[0][0] == self.isciler[j]['name']:
                                        flag = True
                                    else:
                                        j += 1
                                if flag is False:
                                    self.isciler.append({'name': prediction[0][0], 'list': self.checklist})
                                # oteleme yapiliyor
                                prediction = [[name, (top + abs(0 - int(coords[i][0] * frame_height)),
                                                      right + abs(0 - int(coords[i][1] * frame_width)),
                                                      bottom + abs(0 - int(coords[i][0] * frame_height)),
                                                      left + abs(0 - int(coords[i][1] * frame_width)))]
                                              for name, (top, right, bottom, left) in prediction]
                                predictions.append(prediction)
                                # esyalar iscinin box'inin icinde mi kontrolu
                                '''self.isciler[j]['list'] = self.checklistController[1](
                                    detections, self.isciler[j]['list'], coords[i], frame_height, frame_width)'''
                                self.isciler[j]['list'] = self.checklistController[1](
                                    detections, self.checklist, coords[i], frame_height, frame_width)
                    # output = 1
                    self.visualize_and_write(frame, boxes, classes, scores, predictions, output)
                    # Write the resulting image to the output video file
                    print("Writing frame {} / {}".format(frame_number, frames))

                    print(self.isciler)

                    # condition-callback yapisi
                    for captureTuple in self.captureTuples:
                        if captureTuple[0](detections):
                            captureTuple[1](frame)
            self.video.release()
            # output.release()
            cv2.destroyAllWindows()


    def process_frame(self, frame, sess, image_tensor, output_tensors):
        image_expanded = np.expand_dims(frame, axis=0)

        return sess.run(output_tensors, feed_dict={image_tensor: image_expanded})

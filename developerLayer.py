import os
import sys
import cv2

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

import coreLayer


class Checker:
    def __init__(self, checklist, num_frame):
        self.checklist = checklist
        self.num_frame = num_frame

    def on_isci_detection(self, detections):
        th = 0.7
        for i in range(detections['meta']['isci']):
            if detections['objects']['isci'][i][0] >= th:
                return True
        return False

    def is_in_boundary(self, boxes, coords, frame_height, frame_width):
        # box[0] -> y0, box[1] -> x0, box[2] -> y1, box[3] -> x1
        buffer = 10
        # print("boxes: ", boxes[0]*frame_height, boxes[1]*frame_width, boxes[2]*frame_height, boxes[3]*frame_width)
        # print("coords: ", coords[0]*frame_height, coords[1]*frame_width, coords[2]*frame_height, coords[3]*frame_width)
        return \
            boxes[0]*frame_height - buffer <= coords[0]*frame_height and \
            boxes[1]*frame_width - buffer <= coords[1]*frame_width and \
            boxes[2]*frame_height + buffer >= coords[2]*frame_height and \
            boxes[3]*frame_width + buffer >= coords[3]*frame_width

    def is_in_checklist(self, detections, checklist, boxes, frame_height, frame_width):
        th = 0.6
        liste = []
        for element in checklist:
            i = 0
            flag = False
            while i < detections['meta'][element] and flag is False:
                if detections['objects'][element][i][0] >= th:
                    # print("Element: ", element, "Threshold: ", detections['objects'][element][i][0])
                    if self.is_in_boundary(boxes, detections['objects'][element][i][1], frame_height, frame_width):
                        liste.append(element)
                        print("{} element detected".format(element))
                        flag = True
                i += 1
        # benzer elemanlari siler
        checklist = list(set(checklist) ^ set(liste))
        return checklist

    def output_reader(self, outputs):
        print("Outputs : ")
        print(outputs)

    def detect_and_crop(self, detections):
        th = 0.7
        coords = []
        for i in range(detections['meta']['isci']):
            if detections['objects']['isci'][i][0] >= th:
                print(detections['objects']['isci'][i][1])
                coords.append(detections['objects']['isci'][i][1])
        return coords


def on_kask_detection(detections):
    th = 0.7
    for i in range(detections['meta']['kask']):
        if detections['objects']['kask'][i][0] >= th:
            return True
    return False


def callback(frame):
    cv2.imshow("KASKLI_FRAME", frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def on_no_gozluk(detections):
    th = 0.95
    for i in range(detections['meta']['gozluk']):
        if detections['objects']['gozluk'][i][0] < th:
            return True
    return False


def callback2(frame):
    cv2.imshow("GOZLUKSUZ", frame)
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()


def main():
    # 1- Icinde bulundugumuz dizini alir.
    CWD_PATH = os.getcwd()

    # 2- Agirliklari kaydedilmis modelin pathi kaydedilir.
    PATH_TO_CKPT = os.path.join(CWD_PATH, 'inference_graph', 'frozen_inference_graph.pb')

    # 3- Label dosyasının pathi kaydedilir.
    PATH_TO_LABELS = os.path.join(CWD_PATH, 'training', 'labelmap.pbtxt')

    # 4- Label dosyasındaki label sayisi
    NUM_CLASSES = 9

    f_model_path = "/home/zehra/PycharmProjects/face_recognition/trained_knn_model.clf"

    # 5- Kontrol edilmek istenen esyalar burada verilmelidir.
    checker = Checker({'yelek', 'kask', 'gozluk', 'eldiven'}, 2)

    # 7- Core'daki Detection classına modelin agirliklari, label dosyasi, sinif sayisi parametre olarak verilir.
    detectObj = coreLayer.Detection(PATH_TO_CKPT, PATH_TO_LABELS, NUM_CLASSES, f_model_path)

    # 8- Video dosyasının full pathi set_video methoduna parametre olarak verilir.
    detectObj.set_video("/home/zehra/PycharmProjects/object_detection/input9.mp4")

    # 9- Conditionlar ve callback methodları add_conditional_capture
    # isimli methoda verilerek condition ve callback tupleları olusturulur.

    # detectObj.add_conditional_capture(on_kask_detection, callback)
    # detectObj.add_conditional_capture(on_no_gozluk, callback2)

    # 10- Checklist burada coreLayer'a gonderilir.

    detectObj.checklist_controller(checker)

    # 11- Thread calistirilir, developera main threadde calisma olanagi taninir
    detectObj.start()

    print("Main thread doing something else")
    return


if __name__ == '__main__':
    main()
''' 
    ***Agirliklar ve modelin pathi main fonksiyonunda iki numarali adimda verilmelidir.
    ***Labelmap dosyası ve bu dosyanın pathi 3 numaralı adimda verilmelidir.
    ***Labelmap dosyasinin icerdigi label sayisi 4 numarali adimda verilmelidir.
    ***Kontrol edilmek istenen nesneleri iceren checklist 5 numarali adimda verilmelidir.
    ***İsci tespit edildikten sonra kac frame icinde arama yapilmak istedigi 6 numarali adimda verilmelidir.
    ***Video dosyasinin full pathi 8 numarali adimda verilmelidir.
        (Not: Windows kullanicilari icin dosya pathinin direkt kopyalanmasi durumunda Python hata vermektedir. 
        Kopyalanan adres uzerindeki \ isareti "/" ile degistirilmelidir.)
    ***Developer eklemek istedigi condition ve callback fonksiyonlarini main fonksiyonunun uzerinde tanimlanmalidir.
    ***Eklenen fonksiyonlar adim 9'daki gibi add_conditional_capture metoduna parametre olarak verilmelidir.
    ***10 numarali adimda checklist coreLayer'a gonderilir. Arama islemi icin bu adim gereklidir.
    ***11 numarali adim developer tarafindan degistirilmemelidir.
    ***Developer main fonksiyonunun devamina eklemeler yapabilir, core'daki fonksiyonlarin calismasi maini etkilemeyecektir.   
'''

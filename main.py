import sys
import cv2
import numpy
from PIL import Image
from PyQt5 import QtGui
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from keras.models import load_model
from layout import Ui_Dialog
from PyQt5.QtGui import QPixmap

    # Các nhãn của các biển đã train
classes = {1: 'Tốc độ tối đa (20km/h)',
               2: 'Tốc độ tối đa (30km/h)',
               3: 'Tốc độ tối đa (50km/h)',
               4: 'Tốc độ tối đa (60km/h)',
               5: 'Tốc độ tối đa (70km/h)',
               6: 'Tốc độ tối đa (80km/h)',
               7: 'End of speed limit (80km/h)',
               8: 'Tốc độ tối đa (100km/h)',
               9: 'Tốc độ tối đa (120km/h)',
               10: 'Không được vượt',
               11: 'No passing veh over 3.5 tons',
               12: 'Right-of-way at intersection',
               13: 'Đường ưu tiên',
               14: 'Nhường đường',
               15: 'Dừng lại',
               16: 'No vehicles',
               17: 'Veh > 3.5 tons prohibited',
               18: 'Không vào   ',
               19: 'Cẩn thận',
               20: 'Chỗ ngoặt nguy hiểm vòng bên trái',
               21: 'Chỗ ngoặt nguy hiểm vòng bên phải',
               22: 'Double curve',
               23: 'Đường gập ghềnh',
               24: 'Đường trơn trượt',
               25: 'Road narrows on the right',
               26: 'Đường đang thi công',
               27: 'Biển báo giao thông',
               28: 'Pedestrians',
               29: 'Trẻ em qua đường',
               30: 'Bicycles crossing',
               31: 'Beware of ice/snow',
               32: 'Wild animals crossing',
               33: 'End speed + passing limits',
               34: 'Rẽ phải phía trước',
               35: 'Rẽ trái phía trước',
               36: 'Đi thẳng',
               37: 'Go straight or right',
               38: 'Go straight or left',
               39: 'Keep right',
               40: 'Keep left',
               41: 'Roundabout mandatory',
               42: 'End of no passing',
               43: 'End no passing veh > 3.5 tons'}

class MainWindow:
    def __init__(self):
        self.main_win = QMainWindow()
        self.uic = Ui_Dialog()  # lấy giao diện ben
        self.uic.setupUi(self.main_win)
        self.uic.pushButton.clicked.connect(self.click_close)
        self.uic.capture.clicked.connect(self.upload_capture)
       # self.uic.Video.clicked.connect(self.upload_video)
        self.uic.camera.clicked.connect(self.upload_camera)

    def show(self):
        self.main_win.show()

    def click_close(self):
        exit(0)

    def upload_capture(self):
        path_file, _ = QFileDialog.getOpenFileName(None, 'Chọn tệp ảnh', '', 'Images (*.png *.jpg)')
        # Hiển thị ảnh ra màn hình
        self.uic.screen.setPixmap(QPixmap(path_file))
        # Hiển thị nút phân loại ảnh
        self.show_classify_capture(path_file)

    def show_classify_capture(self,path_file):
        model = load_model('traffic_classifier.h5')
        image = Image.open(path_file)
        image = image.resize((30, 30))
        image = numpy.expand_dims(image, axis=0)
        image = numpy.array(image)
        predictions = model.predict(image) # tạo ra 1 tập các dự đoán của image
        pred = numpy.argmax(predictions, axis=1)  # tìm chỉ số của lớp có xác suất dự đoán cao nhất trên mỗi hàng của ma trận predictions.
        prob = numpy.amax(pred) #tìm chỉ giá trị lớn nhất trong mảng các chỉ số này
        prob1 = numpy.amax(predictions) #tìm giá trị xác xuất cao nhất
        sign = classes[prob + 1]
        print(prob1)
        self.uic.result.setText(sign)

    def upload_camera(self):
        model = load_model('model.h5')
        file = 'stop.mp4'
        self.cap = cv2.VideoCapture(0)
        while True:
            # Lấy hình ảnh từ webcam
            OK, self.frame = self.cap.read()
            # Nếu nhấn phím 'q' thì thoát khỏi vòng lặp
            img = numpy.asarray(self.frame)
            img = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
            img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_AREA)
            mp = QtGui.QImage(img, img.shape[1], img.shape[0], img.strides[0], QtGui.QImage.Format_Grayscale8)
            #xử lí trên màn hình chính
            img1 = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            img1 = cv2.resize(img1, (778, 510), interpolation=cv2.INTER_AREA)
            mp1 = QtGui.QImage(img1, img1.shape[1], img1.shape[0], img1.strides[0], QtGui.QImage.Format_RGB888)
            self.uic.screen.setPixmap(QtGui.QPixmap.fromImage(mp1))


            mp = img / 255.0
            mp = mp.reshape(1, 32, 32, 1)
            predictions = model.predict(mp)
            pred = numpy.argmax(predictions, axis=1)
            prob = numpy.amax(pred)
            prob1 = numpy.amax(predictions)
            sign = classes[prob + 1]
            if prob1 > 0.7:
                print(prob1)
                self.uic.result.setText(sign)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())

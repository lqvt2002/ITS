def upload_camera(self):
    model = load_model('traffic_classifier.h5')
    filepm4 = 'testmp4.mp4'
    self.cap = cv2.VideoCapture(0)
    while True:
        # Lấy hình ảnh từ webcam
        OK, self.frame = self.cap.read()
        # Nếu nhấn phím 'q' thì thoát khỏi vòng lặp
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
        test = gray  # lưu mp4 vào biến tạm mp1 để xử lý nhận diện ở phía sau

        mp4 = cv2.resize(gray, (778, 510), interpolation=cv2.INTER_AREA)
        mp = QtGui.QImage(mp4, mp4.shape[1], mp4.shape[0], mp4.strides[0], QtGui.QImage.Format_RGB888)
        self.uic.screen.setPixmap(QtGui.QPixmap.fromImage(mp))

        mp1 = cv2.resize(test, (30, 30), interpolation=cv2.INTER_AREA)
        mp = mp1 / 255.0  # chuẩn hóa giá trị pixel
        mp = numpy.expand_dims(mp, axis=0)
        mp = numpy.array(mp)
        predictions = model.predict(mp)
        pred = numpy.argmax(predictions, axis=1)
        prob = numpy.amax(pred)
        prob1 = numpy.amax(predictions)
        sign = classes[prob + 1]
        if prob1 > 0.5:
            print(prob1)
            self.uic.result.setText(sign)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
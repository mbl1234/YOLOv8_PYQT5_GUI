import sys
import cv2
from ultralytics.yolo.engine.model import YOLO
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import *
import ui_img.detect_images_rc
class Ui_MainWindow(QMainWindow):
    def __init__(self):
        super(Ui_MainWindow, self).__init__()
        self.setWindowTitle("基于YOLOv8的检测演示软件V1.0")
        self.resize(1500, 1000)
        self.setStyleSheet("QWidget#centralwidget{background-image: url(:/detect_background/detect.JPG);}")
        self.centralwidget = QWidget()
        self.centralwidget.setObjectName("centralwidget")

        # 模型选择        
        self.btn_selet_model = QtWidgets.QPushButton(self.centralwidget)
        self.btn_selet_model.setGeometry(QtCore.QRect(70, 810, 70, 70))
        self.btn_selet_model.setStyleSheet("border-image: url(:/detect_button_background/upload.png);")
        self.btn_selet_model.setText("")
        self.btn_selet_model.setObjectName("btn_selet_model")
        self.btn_selet_model.clicked.connect(self.seletModels)

        # 选择图像进行检测
        self.btn_detect_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_detect_img.setGeometry(QtCore.QRect(390, 810, 70, 70))
        self.btn_detect_img.setStyleSheet("border-image: url(:/detect_button_background/images.png);")
        self.btn_detect_img.setText("")
        self.btn_detect_img.setObjectName("btn_detect_img")
        self.btn_detect_img.clicked.connect(self.openImage)
        
        # 保存结果图像
        self.btn_save_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_save_img.setGeometry(QtCore.QRect(730, 810, 70, 70))
        self.btn_save_img.setStyleSheet("border-image: url(:/detect_button_background/save.png);")
        self.btn_save_img.setText("")
        self.btn_save_img.setObjectName("btn_save_img")
        self.btn_save_img.clicked.connect(self.saveImage)

        # 清除结果图像
        self.btn_clear_img = QtWidgets.QPushButton(self.centralwidget)
        self.btn_clear_img.setGeometry(QtCore.QRect(1050, 810, 70, 70))
        self.btn_clear_img.setStyleSheet("border-image: url(:/detect_button_background/delete.png);")
        self.btn_clear_img.setText("")
        self.btn_clear_img.setObjectName("btn_clear_img")
        self.btn_clear_img.clicked.connect(self.clearImage)

        # 退出应用
        self.btn_exit_app = QtWidgets.QPushButton(self.centralwidget)
        self.btn_exit_app.setGeometry(QtCore.QRect(1360, 810, 70, 70))
        self.btn_exit_app.setStyleSheet("border-image: url(:/detect_button_background/exit.png);")
        self.btn_exit_app.setText("")
        self.btn_exit_app.setObjectName("btn_exit_app")
        self.btn_exit_app.clicked.connect(self.exitApp)

        # 呈现原始图像
        self.label_show_yuanshi = QtWidgets.QLabel(self.centralwidget)
        self.label_show_yuanshi.setGeometry(QtCore.QRect(0, 80, 700, 700))
        self.label_show_yuanshi.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_yuanshi.setObjectName("label_show_yuanshi")
        
        # 呈现结果图像
        self.label_show_jieguo = QtWidgets.QLabel(self.centralwidget)
        self.label_show_jieguo.setGeometry(QtCore.QRect(800, 80, 700, 700))
        self.label_show_jieguo.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_jieguo.setObjectName("label_show_jieguo")

        # 呈现功能按键
        self.label_show_button = QtWidgets.QLabel(self.centralwidget)
        self.label_show_button.setGeometry(QtCore.QRect(0, 800, 1501, 141))
        self.label_show_button.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.label_show_button.setText("")
        self.label_show_button.setObjectName("label_show_button")

        #编写模型加载
        self.edit_selet_model = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_selet_model.setGeometry(QtCore.QRect(20, 890, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(28)
        self.edit_selet_model.setFont(font)
        self.edit_selet_model.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.edit_selet_model.setObjectName("edit_selet_model")

        #编写图像加载
        self.edit_detect_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_detect_img.setGeometry(QtCore.QRect(350, 890, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(28)
        self.edit_detect_img.setFont(font)
        self.edit_detect_img.setObjectName("edit_detect_img")
        
        #编写图像保存
        self.edit_save_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_save_img.setGeometry(QtCore.QRect(690, 890, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(28)
        self.edit_save_img.setFont(font)
        self.edit_save_img.setObjectName("edit_save_img")
        
        #编写图像清除
        self.edit_clear_img = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_clear_img.setGeometry(QtCore.QRect(1000, 890, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(28)
        self.edit_clear_img.setFont(font)
        self.edit_clear_img.setObjectName("edit_clear_img")
        
        #编写应用退出
        self.edit_exit_app = QtWidgets.QLineEdit(self.centralwidget)
        self.edit_exit_app.setGeometry(QtCore.QRect(1300, 890, 161, 40))
        font = QtGui.QFont()
        font.setFamily("Adobe 宋体 Std L")
        font.setPointSize(28)
        self.edit_exit_app.setFont(font)
        self.edit_exit_app.setObjectName("edit_exit_app")

        # 标题
        self.label_show_title = QtWidgets.QLabel(self.centralwidget)
        self.label_show_title.setGeometry(QtCore.QRect(190, 10, 1101, 80))
        font = QtGui.QFont()
        font.setFamily("Adobe 黑体 Std R")
        font.setPointSize(28)
        self.label_show_title.setFont(font)
        self.label_show_title.setStyleSheet("")
        self.label_show_title.setObjectName("label_show_title")

        self.label_show_button.raise_()
        self.btn_selet_model.raise_()
        self.btn_detect_img.raise_()
        self.btn_save_img.raise_()
        self.btn_clear_img.raise_()
        self.btn_exit_app.raise_()
        self.label_show_title.raise_()
        self.label_show_yuanshi.raise_()
        self.label_show_jieguo.raise_()
        self.edit_selet_model.raise_()
        self.edit_detect_img.raise_()
        self.edit_save_img.raise_()
        self.edit_clear_img.raise_()
        self.edit_exit_app.raise_()

        # 主窗口
        self.setCentralWidget(self.centralwidget)
        self.retranslateUi(self.centralwidget)
        QtCore.QMetaObject.connectSlotsByName(self.centralwidget)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_show_title.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:28pt; font-weight:600; color:#ffffff;\">基于YOLOv8的检测演示软件</span></p></body></html>"))
        self.label_show_yuanshi.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">原始图像</span></p></body></html>"))
        self.label_show_jieguo.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:20pt;\">检测图像</span></p></body></html>"))
        self.edit_selet_model.setText(_translate("MainWindow", "模型加载"))
        self.edit_detect_img.setText(_translate("MainWindow", "图像加载"))
        self.edit_save_img.setText(_translate("MainWindow", "图像保存"))
        self.edit_clear_img.setText(_translate("MainWindow", "图像清除"))
        self.edit_exit_app.setText(_translate("MainWindow", "应用退出"))

    def seletModels(self):
        self.openfile_name_model, _ = QFileDialog.getOpenFileName(self.btn_selet_model, '选择weights文件', '.', '权重文件(*.pt)')
        if not self.openfile_name_model:
            QMessageBox.warning(self, "Warning:", "打开权重失败", buttons=QMessageBox.Ok,)
        else:
            print('加载weights文件地址为：' + str(self.openfile_name_model))
            QMessageBox.information(self, u"Notice", u"权重打开成功", buttons=QtWidgets.QMessageBox.Ok)

    def openImage(self):
        print(1)
        name_list = []
        print(2)
        fname, _ = QFileDialog.getOpenFileName(self, '打开文件', '.', '图像文件(*.jpg)')
        print(3)
        self.fname = fname
        print(4)
        pixmap = QtGui.QPixmap(fname)
        print(4.1)
        self.label_show_yuanshi.setPixmap(pixmap)
        print(5)
        self.label_show_yuanshi.setScaledContents(True)
        print(6)
        img = cv2.imread(fname)
        print(7)
        model = YOLO(self.openfile_name_model)
        print(8)
        results = model.predict(source=self.fname)
        print(9)
        annotated_frame = results[0].plot()
        print(10)
        #方法二：
        # 将图像数据转换为QImage格式
        height, width, channel = annotated_frame.shape
        print(11)
        bytes_per_line = 3 * width
        print(12)
        qimage = QtGui.QImage(annotated_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888)
        print(13)
        self.qImg = qimage
        print(13.1)
        # 将QImage转换为QPixmap
        pixmap = QtGui.QPixmap.fromImage(qimage)
        print(14)
        self.label_show_jieguo.setPixmap(pixmap)
        print(15)
        self.label_show_jieguo.setScaledContents(True)
        print(16)
        return self.qImg
        print(17)


    def saveImage(self):
        fd, _ = QFileDialog.getSaveFileName(self, "保存图片", ".", "*.jpg")
        self.qImg.save(fd)

    def clearImage(self, stopp):
        result = QMessageBox.question(self, "Warning:", "是否清除本次检测结果", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            self.label_show_yuanshi.clear()
            self.label_show_jieguo.clear()
        else:
            stopp.ignore()

    def exitApp(self, event):
        event = QApplication.instance()
        result = QMessageBox.question(self, "Notice:", "您真的要退出此应用吗", QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
        if result == QMessageBox.Yes:
            event.quit()
        else:
            event.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    ui = Ui_MainWindow()
    ui.show()
    sys.exit(app.exec_())
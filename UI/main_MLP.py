import sys
from PyQt5.QtWidgets import QApplication, QMainWindow
# 导入my_win.py中内容
from UI.newMLP import *
# 创建mainWin类并传入Ui_MainWindow


class mainWin(QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        super(mainWin, self).__init__(parent)
        self.setupUi(self)
        self.initialize()

if __name__ == '__main__':

    app = QApplication(sys.argv)
    main_win = mainWin()
    main_win.show()
    sys.exit(app.exec_())
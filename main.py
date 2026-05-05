import sys
import os

# venv 내 Qt 플랫폼 플러그인 경로를 명시적으로 지정
# PyQt5 import 시점에 플랫폼 플러그인을 로드하므로, import 전에 반드시 설정해야 한다
_qt_plugins = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "PyQt5", "Qt5", "plugins")
os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.normpath(_qt_plugins))

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from gui.main_window import MainWindow


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setApplicationName("CCTV 이상 현상 감지 시스템")
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()

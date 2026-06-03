import sys
import os

# venv 내 Qt 플랫폼 플러그인 경로를 명시적으로 지정
# PyQt5 import 시점에 플랫폼 플러그인을 로드하므로, import 전에 반드시 설정해야 한다

# 1) 윈도우(Windows)용
if sys.platform == "win32":
    _qt_plugins = os.path.join(os.path.dirname(sys.executable), "..", "Lib", "site-packages", "PyQt5", "Qt5", "plugins")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.normpath(_qt_plugins))

# 2) 맥(Mac)용
elif sys.platform == "darwin":
    # Mac 환경에서는 보통 자동으로 로드되므로 주석 처리합니다.
    # 만약 플러그인 관련 에러가 발생하면, 사용하는 파이썬 버전에 맞춰 아래 주석을 해제하세요. (예: python3.11)
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    _qt_plugins = os.path.join(os.path.dirname(sys.executable), "..", "lib", python_version, "site-packages", "PyQt5", "Qt5", "plugins")
    os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.normpath(_qt_plugins))
    pass

# 3) 라즈베리파이(Linux)용
elif sys.platform == "linux":
    # 라즈베리파이(Linux) 환경에서도 주로 자동 인식되므로 주석 처리합니다.
    # 에러 발생 시 시스템 환경에 맞게 주석을 해제하여 사용하세요.
    # _qt_plugins = os.path.join(os.path.dirname(sys.executable), "..", "lib", "python3.11", "site-packages", "PyQt5", "Qt5", "plugins")
    # os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", os.path.normpath(_qt_plugins))
    pass

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

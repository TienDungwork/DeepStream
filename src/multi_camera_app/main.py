import sys
import os
import signal

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from multi_camera_app.controllers import AppController
signal.signal(signal.SIGINT, signal.SIG_DFL)


def main():
    controller = AppController()
    controller.run()


if __name__ == '__main__':
    sys.exit(main())

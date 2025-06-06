
from roboflow import Roboflow
rf = Roboflow(api_key="oqErh7VvYQFESAMtK5eU")
project = rf.workspace("fruitsdetection").project("fruits-by-yolo")
version = project.version(1)
dataset = version.download("yolov8")
                
                
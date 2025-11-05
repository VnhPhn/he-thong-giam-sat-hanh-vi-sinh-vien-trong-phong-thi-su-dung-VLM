
from roboflow import Roboflow
rf = Roboflow(api_key="hq4P8wCLO9Ye4AcveVvZ")
project = rf.workspace("proctoring-pjt1g").project("proctoring-obj")
version = project.version(2)
dataset = version.download("yolov8")
                
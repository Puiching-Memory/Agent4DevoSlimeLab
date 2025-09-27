from ultralytics import YOLO

if "__main__" == __name__:
    model = YOLO("runs/slime_yolo/loop2/weights/best.pt")
    model.export(format="onnx",
                 opset=21)
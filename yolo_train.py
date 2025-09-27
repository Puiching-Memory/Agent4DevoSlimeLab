from ultralytics import YOLO

if "__main__" == __name__:
    # Load a pretrained YOLO model (recommended for training)

    #model = YOLO("yolo11n.pt")
    model = YOLO("yolo11m.pt")

    # Train the model using the dataset configuration file
    # https://docs.ultralytics.com/zh/modes/train/#train-settings
    results = model.train(data="data.yaml",
                          project="runs/slime_yolo",
                          exist_ok=False,
                          epochs=100,
                          imgsz=640,
                          workers=1,
                          compile=False,
                          )
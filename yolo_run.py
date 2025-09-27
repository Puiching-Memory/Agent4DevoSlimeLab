from ultralytics import YOLO

if "__main__" == __name__:
    model = YOLO("runs/slime_yolo/loop3/weights/best.pt")
    
    # run and save as vedio
    model.predict(source="QQ2025927-173336.mp4",
                  save=True,
                  half=True,
                  batch=4,
                  show=True,
                  )
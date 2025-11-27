from ultralytics import YOLO

def main():
    # Load a COCO-pretrained YOLO12n model
    model = YOLO("yolo12n.pt")

    # Train the model on the COCO8 example dataset for 10 epochs
    results = model.train(
        data="./aortic_valve.yaml",
        epochs=50,
        imgsz=640,
        batch=4,
        device=0
    )

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()

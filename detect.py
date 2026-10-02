"""Live laser-dot detection from a webcam, RTSP stream, video file or image.

Runs the DVC-tracked YOLO26n models on this machine (Apple GPU via MPS when available).

Examples:
    python detect.py                                   # built-in webcam, green model
    python detect.py --laser red                       # red model
    python detect.py --laser both                      # run red and green models together
    python detect.py --source 1                        # another camera (e.g. iPhone Continuity Camera)
    python detect.py --source rtsp://user:pass@192.168.1.20:554/stream1
    python detect.py --source clip.mp4 --save out.mp4
    python detect.py --source data/raw/green/test/images/<frame>.jpg

Keys in the window: q / Esc quits, s saves a snapshot.
"""

import argparse
import sys
import threading
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))
from device import best_device  # noqa: E402

MODELS = {"red": "models/released/red_yolo26n.pt", "green": "models/released/green_yolo26n.pt"}
# Default confidence per laser. Red gives false alarms on webcam scenes at 0.25; at 0.80 the
# red test set has 0 false alarms and 8/113 misses (vs 1 and 3 at 0.25).
CONF = {"red": 0.80, "green": 0.25}
COLORS = {"red": (0, 0, 255), "green": (0, 255, 0)}  # BGR
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LatestFrame:
    """Reads a live stream on a background thread and keeps only the newest frame,
    so slow inference never builds up lag on RTSP / webcam sources."""

    def __init__(self, cap):
        self.cap, self.frame, self.ok = cap, None, True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        while self.ok:
            ok, frame = self.cap.read()
            with self.lock:
                self.ok, self.frame = ok, frame if ok else self.frame

    def read(self):
        with self.lock:
            return self.ok, None if self.frame is None else self.frame.copy()


def open_source(source):
    if source.isdigit():
        # AVFoundation is the native macOS camera backend
        backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
        cap = cv2.VideoCapture(int(source), backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        return cap, True
    if source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://")):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap, True
    return cv2.VideoCapture(source), False  # video file: read every frame


def detect(models, frame, args):
    """Returns [(laser, conf, x1, y1, x2, y2), ...] for every model."""
    found = []
    for laser, model in models.items():
        r = model.predict(frame, conf=args.conf or CONF[laser], imgsz=args.imgsz, device=args.device, verbose=False)[0]
        for box, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
            found.append((laser, conf, *map(int, box)))
    return found


def draw(frame, found, fps=None):
    for laser, conf, x1, y1, x2, y2 in found:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        color = COLORS[laser]
        cv2.rectangle(frame, (x1 - 6, y1 - 6), (x2 + 6, y2 + 6), color, 2)
        cv2.circle(frame, (cx, cy), 18, color, 1)
        cv2.putText(frame, f"{laser} {conf:.2f} ({cx},{cy})", (x1, max(y1 - 12, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    status = f"LASER x{len(found)}" if found else "no laser"
    if fps is not None:
        status += f"  |  {fps:.1f} FPS"
    cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(frame, status, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1)
    return frame


def run_image(models, args):
    frame = cv2.imread(args.source)
    found = detect(models, frame, args)
    for d in found:
        print(f"{d[0]} conf={d[1]:.2f} box=({d[2]},{d[3]},{d[4]},{d[5]})")
    print(f"{len(found)} laser(s) found")
    out = draw(frame, found)
    if args.save:
        cv2.imwrite(args.save, out)
    if not args.no_show:
        cv2.imshow("laser detector", out)
        cv2.waitKey(0)


def run_stream(models, args):
    cap, live = open_source(args.source)
    if not cap.isOpened():
        sys.exit(f"Could not open source {args.source!r}. For the webcam on macOS, allow camera access for "
                 "your terminal app in System Settings > Privacy & Security > Camera.")
    reader = LatestFrame(cap) if live else None
    writer, fps, last, last_print = None, 0.0, time.time(), 0.0

    while True:
        ok, frame = reader.read() if reader else cap.read()
        if not ok:
            break
        if frame is None:  # live stream has not delivered its first frame yet
            time.sleep(0.01)
            continue

        found = detect(models, frame, args)
        now = time.time()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-6) if fps else 1 / max(now - last, 1e-6)
        last = now

        if found and now - last_print > 0.5:  # print at most twice a second
            best = max(found, key=lambda d: d[1])
            print(f"[{time.strftime('%H:%M:%S')}] {best[0]} laser at "
                  f"({(best[2] + best[4]) // 2},{(best[3] + best[5]) // 2}) conf={best[1]:.2f}")
            last_print = now

        out = draw(frame, found, fps)
        if args.save:
            if writer is None:
                h, w = out.shape[:2]
                writer = cv2.VideoWriter(args.save, cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))
            writer.write(out)
        if not args.no_show:
            cv2.imshow("laser detector", out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                name = f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(name, out)
                print(f"saved {name}")

    if reader:
        reader.ok = False
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="0", help="camera index, rtsp:// URL, video file or image (default 0)")
    p.add_argument("--laser", choices=["green", "red", "both"], default="green")
    p.add_argument("--conf", type=float, default=None,
                   help="confidence threshold for every model (default: red 0.80, green 0.25)")
    p.add_argument("--imgsz", type=int, default=1280,
                   help="inference size; the models were trained at 1280. 960/640 is faster but misses small dots")
    p.add_argument("--device", default=None, help="mps | cpu | 0 (default: best available)")
    p.add_argument("--save", help="write the annotated output to this file")
    p.add_argument("--no-show", action="store_true", help="do not open a window")
    args = p.parse_args()
    args.device = args.device or best_device()

    from ultralytics import YOLO
    lasers = ["red", "green"] if args.laser == "both" else [args.laser]
    models = {}
    for laser in lasers:
        path = Path(__file__).parent / MODELS[laser]
        if not path.exists():
            sys.exit(f"{path} is missing - run `dvc pull` first.")
        models[laser] = YOLO(str(path))
    confs = ", ".join(f"{l} {args.conf or CONF[l]:.2f}" for l in lasers)
    print(f"models: {', '.join(lasers)} | conf: {confs} | device: {args.device} | imgsz: {args.imgsz} | source: {args.source}")

    if Path(args.source).suffix.lower() in IMAGE_EXT:
        run_image(models, args)
    else:
        run_stream(models, args)


if __name__ == "__main__":
    main()

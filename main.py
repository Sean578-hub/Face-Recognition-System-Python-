import cv2
import face_recognition
import mss
import numpy as np
import os

PEOPLE_DIR = "People"
TOLERANCE = 0.45

def load_known_faces(people_dir = PEOPLE_DIR):
    known_encodings = []
    known_names = []

    for fname in sorted(os.listdir(people_dir)):
        path = os.path.join(people_dir, fname)
        img = face_recognition.load_image_file(path)
        encs = face_recognition.face_encodings(img)
        if encs:
            base = os.path.splitext(fname)[0]
            name = "".join([c for c in base if not c.isdigit()])
            known_encodings.append(encs[0])
            known_names.append(name)


    return known_encodings, known_names

def distance_to_confidence(d, tol = TOLERANCE):
    conf = 1.0 - (d / max(tol, 1e-6))
    return float(np.clip(conf, 0.0, 1.0))

def main():
    known_encodings, known_names = load_known_faces()

    sct = mss.mss()
    info_monitor = sct.monitors[1]
    monitor = {"top": info_monitor["top"], "left": info_monitor["left"], "width": 1500, "height": 1230}

    while True:

        screenshare = sct.grab(monitor)
        imgs = np.array(screenshare)
        screenshare_bgr = cv2.cvtColor(imgs, cv2.COLOR_BGRA2BGR)

        names, boxes, confs = [], [], []


        rgb = cv2.cvtColor(screenshare_bgr, cv2.COLOR_BGR2RGB)
        location = face_recognition.face_locations(rgb, model = "hog")
        encoding = face_recognition.face_encodings(rgb, location)
        for (face_encodings, (top, right, bottom, left)) in zip(encoding, location):
            if known_encodings:
                dists = face_recognition.face_distance(known_encodings, face_encodings)
                best_i = int(np.argmin(dists))
                best_d = float(dists[best_i])
                if best_d <= TOLERANCE:
                    name = known_names[best_i]
                    conf = distance_to_confidence(best_d)
                else:
                    name, conf = "Unknown", 0.0
            else:
                name, conf = "Unknown", 0.0
            names.append(name)
            boxes.append((top, right, bottom, left))
            confs.append(conf)
        for (t, r, b, l), name, conf in zip(boxes, names, confs):
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)

            cv2.rectangle(screenshare_bgr, (l, t), (r, b), color, 2)

            label = name if name == "Unknown" else f"{name} {int(conf * 100)}%"
            (tw, th), bl = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            y1 = max(t - th - 8, 0)
            cv2.rectangle(screenshare_bgr, (l, y1), (l + tw + 8, y1 + th + bl + 6), color, -1)
            cv2.putText(screenshare_bgr, label, (l + 4, y1 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow("Screenshare", screenshare_bgr)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

Face Recognition System (Python)

This project detects and identifies faces in real time from a webcam feed.
It can recognize multiple people simultaneously using image encoding.

Technologies:
Python
OpenCV
face_recognition library
NumPy











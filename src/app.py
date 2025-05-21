import cv2
import csv
import time
from PytorchWildlife.models import detection as pw_detection
from deep_sort_realtime.deepsort_tracker import DeepSort


camera_zones = {
    0: 'Cheeku Forest Path',
    "data/sample_video.mp4": 'South Waterhole'
}


def run_detection_on_source(source, zone, max_frames=None):

    detector = pw_detection.MegaDetectorV6(version='MDV6-yolov9-c')
    tracker = DeepSort(max_age = 10)

    cap = cv2.VideoCapture(source)
    csv_file = open(f'data/detections_log_{zone.replace(" ","_")}.csv', mode='w', newline='')
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(['FRAME', 'TIME', 'ZONE', 'TRACK', 'CLASS', 'CONF', 'x1', 'y1', 'x2', 'y2'])

    frame_count = 0

    while cap.isOpened(): 

        ret, frame = cap.read()
        if not ret or (max_frames and frame_count >= max_frames):
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.single_image_detection(frame_rgb)

        detections = []
        for det in results['detections']: 
            bbox_tuple, conf, category_idx, *_ = det
            x1, y1, x2, y2 = bbox_tuple
            label = detector.CLASS_NAMES[int(category_idx)]  
            w, h = x2 - x1, y2 - y1
            bbox = [x1, y1, w, h]
            detections.append((
                [x1, y1, w, h],  
                float(conf),     
                int(category_idx) 
            ))

        tracks = tracker.update_tracks(detections, frame=frame_rgb)

        timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

        for track in tracks: 
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            class_id = track.det_class 
            ltrb = track.to_ltrb()
            conf = track.det_conf
            x1, y1, x2, y2 = map(int, ltrb)

            csv_writer.writerow([frame_count, timestamp, zone, track_id, class_id, conf, x1, y1, x2, y2])

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, f'ID:{track_id}', (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        cv2.imshow(f'{zone} - Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break    

        frame_count += 1

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Detection log saved for zone: '{zone}'")

for source, zone in camera_zones.items():
    run_detection_on_source(source, zone, max_frames=100)

from paddleocr import PaddleOCR, draw_ocr
import cv2
import numpy as np

# Initialize OCR model
ocr = PaddleOCR(use_angle_cls=True, lang='en')  # CPU optimized English model

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

frame_count = 0
ocr_results = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_count += 1

    # Define ROI (Region Of Interest) in center
    roi = frame[150:400, 220:420]

    if frame_count % 10 == 0:  # OCR every 10 frames
        ocr_results = ocr.ocr(roi, cls=True)

    # Draw ROI box (for visual guide)
    cv2.rectangle(frame, (220, 150), (420, 400), (255, 255, 0), 2)

    # Draw OCR results
    if ocr_results:
        for line in ocr_results[0]:
            box = line[0]
            text = line[1][0]
            confidence = line[1][1]

            box = [tuple(map(int, point)) for point in box]
            shifted_box = [(pt[0] + 220, pt[1] + 150) for pt in box]

            cv2.polylines(frame, [np.array(shifted_box)], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.putText(frame, text, shifted_box[0], cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('PaddleOCR Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

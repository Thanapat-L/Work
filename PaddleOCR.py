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

    #frame = cv2.flip(frame, 1)
    frame_count += 1

    # Define ROI (Region Of Interest) in center
    #roi = frame[150:400, 220:420]

    if frame_count % 10 == 0:  # OCR every 10 frames
        ocr_results = ocr.ocr(frame, cls=True)

    # Draw ROI box (for visual guide)
    #cv2.rectangle(frame, (220, 150), (420, 400), (255, 255, 0), 2)

    # Draw OCR results
    if ocr_results and isinstance(ocr_results, list) and isinstance(ocr_results[0], list):
        if len(ocr_results[0]) > 0:
            for line in ocr_results[0]:
                box = line[0]
                text = line[1][0]
                confidence = line[1][1]

                box = [tuple(map(int, point)) for point in box]

                # Find center of box to put text nicely
                x_coords = [p[0] for p in box]
                y_coords = [p[1] for p in box]
                center_x = int(sum(x_coords) / len(x_coords))
                center_y = int(sum(y_coords) / len(y_coords))

                (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

                # Draw filled rectangle behind the text
                cv2.rectangle(frame, (center_x, center_y - text_h), (center_x + text_w, center_y), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, text, (center_x, center_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imshow('PaddleOCR Webcam', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

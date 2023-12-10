import cv2
import pytesseract

from utils.utils import process_image, collect_key_word_positions, apply_bounding_boxes

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

KEY_WORD = "technology"

# Settings
width = 1920
height = 1080

# Obtaining web camera
cap = cv2.VideoCapture(0)
# cap.set(widthIndex, width)
cap.set(3, width)
# cap.set(heightIndex, height)
cap.set(4, height)

while True:

    success, img = cap.read()

    # if something:
    height, width, *rest = img.shape

    data = process_image(img)

    key_word_positions = collect_key_word_positions(data, KEY_WORD)
    img = apply_bounding_boxes(data, img, key_word_positions)

    cv2.imshow("img", img)
    # "q" key for exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

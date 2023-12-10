import cv2
import pytesseract
from pytesseract import Output

PT_CONFIG = r"--psm 11 --oem 3"


def process_image(img):
    return pytesseract.image_to_data(img, config=PT_CONFIG, output_type=Output.DICT)


def collect_key_word_positions(data, key_word):
    key_word = key_word.lower()
    key_word_positions = []
    current_index = 0
    for word in data["text"]:
        if (key_word in str(word).strip().lower()):
            key_word_positions.append(current_index)
        current_index += 1
    return key_word_positions


def apply_bounding_boxes(data, img, key_word_positions):
    for index in key_word_positions:
        if float(data["conf"][index] > 90):
            (x, y, width, height) = (
                data["left"][index], data["top"][index], data["width"][index], data["height"][index])
            img = cv2.rectangle(img, (x, y), (x + width, y + height), (0, 255, 0), 2)
    return img

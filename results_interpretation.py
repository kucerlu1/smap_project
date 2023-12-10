import os

import cv2
import pytesseract

from utils.utils import process_image, collect_key_word_positions

DATA_FOLDER = "./data/Dive_into_OCR_images"

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

KEY_WORDS = ["locate", "image", "process", "document", "communication", "technologies", "application", "real-time", "structure", "neural"]

KEY_WORDS_MAP = {x: 0 for x in KEY_WORDS}

print("Total image count: ", len(os.listdir(DATA_FOLDER)))
for filename in os.listdir(DATA_FOLDER):
    print("Loading image: " + filename)
    img = cv2.imread(os.path.join(DATA_FOLDER, filename))

    if img is not None:
        data = process_image(img)

        for key_word in KEY_WORDS:
            appearance = collect_key_word_positions(data, key_word)
            KEY_WORDS_MAP[key_word] = KEY_WORDS_MAP[key_word] + len(appearance)

print(KEY_WORDS_MAP)

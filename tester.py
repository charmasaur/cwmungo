import sys
import base64
import cwmungo.crossword_extractor as crossword_extractor
import cv2
import numpy as np

if len(sys.argv) < 2:
    print("Usage: python tester.py <filename> [-s]")
    sys.exit()

filename = sys.argv[1]
data = open(filename, "rb").read()
b64data = base64.b64encode(data)

crossword_extractor.set_debug()
output = crossword_extractor.apply({"b64data":b64data})

# Print the basic output
sys.stdout.write(output)
sys.stdout.flush()

if len(sys.argv) > 2 and sys.argv[2] == "-s":
    lines = output.split("|")
    width, height = map(int, lines[0].split(" "))

    # And save the crossword if necessary
    DIM = 30
    output_image = np.zeros((height * DIM, width * DIM, 1), np.uint8)
    for (r, line) in enumerate(lines[1:]):
        for (c, ch) in enumerate(line):
            if ch == ' ':
                cv2.rectangle(output_image, (c * DIM, r * DIM), (c * DIM + DIM - 2, r * DIM + DIM - 2), 255, -1)

    cv2.imwrite("tmp/tmp.png", output_image)

sys.exit()

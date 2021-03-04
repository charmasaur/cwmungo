import base64
import cv2
import numpy as np
import math

debug = False

def set_debug():
    global debug
    debug = True

show_count = 0
# helper to display an image if debug is true
def show(input):
    global show_count
    if not debug:
        return
    cv2.imwrite("tmp/tmp" + str(show_count) + ".png", input)
    show_count = show_count + 1

# helper to convert an angle (in radians) to [-pi, pi)
def principal_angle(angle):
    tmp = math.fmod(angle, 2 * math.pi) # (-2pi, 2pi)
    if tmp < 0:
        tmp += 2 * math.pi
    # [0, 2pi)
    return math.fmod(tmp + math.pi, 2 * math.pi) - math.pi # [-pi, pi)

# helper to threshold an image
def do_threshold(input):
    return cv2.adaptiveThreshold(
            input, 255., cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, 20)

# gets the rotation angle using hough transform (works best with a single big rectangle, like a mask)
def get_angle_hough(input):
    edge = cv2.Canny(input, 50, 200)

    thresh = 0
    while True:
        thresh += 10
        lines = cv2.HoughLines(edge, 5, math.pi/180, thresh)
        # TODO: Failing here (lines is none)
        if len(lines) <= 10:
            break

    angles = 0.
    angle_count = 0
    for thingo in lines:
        rho = thingo[0][0]
        theta = thingo[0][1]
        pang = principal_angle(theta) # (-pi, pi)
        if pang < -math.pi / 2:
            pang += math.pi
        if pang > math.pi / 2:
            pang -= math.pi
        # (-pi/2, pi/2)
        if pang < math.pi / 4 and pang > -math.pi/4:
            angles += pang
            angle_count += 1
        
    rot_angle_rad = angles / float(angle_count)
    rot_angle_deg = rot_angle_rad * 180. / math.pi
    return rot_angle_deg

# helper to rotate by an angle (in degrees)
def rotate(input, angle):
    midr = input.shape[0] / 2
    midc = input.shape[1] / 2
    # Actually rotate the input
    rot = cv2.getRotationMatrix2D((midc, midr), angle, 1.)
    return cv2.warpAffine(input, rot, (input.shape[1], input.shape[0]))

# crossword mask
def get_cw_mask(input):
    filled = input.copy()
    filled = do_threshold(filled)
    # Fill from all corners
    ini = 1

    col = (0, 0, 0)
    mask = np.zeros((filled.shape[0] + 2, filled.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, mask, (ini,ini), col)
    cv2.floodFill(filled, mask, (filled.shape[1] - ini,ini), col)
    cv2.floodFill(filled, mask, (filled.shape[1] - ini,filled.shape[0] - ini), col)
    cv2.floodFill(filled, mask, (ini,filled.shape[0] - ini), col)
    show(mask*255)
    # Find average white pixel
    tr = 0
    tc = 0
    locs = np.nonzero(filled)
    nlocs = len(locs[0])
    for i in range(nlocs):
      tc += locs[1][i]
      tr += locs[0][i]
    center = (int(float(tc) / float(nlocs)), int(float(tr) / float(nlocs)))

    oldmask = mask.copy()

    bc = (255, 255, 255)
    cv2.floodFill(filled, mask, center, (255, 0, 0), bc, bc)
    mask -= oldmask

    show(mask*255)

    # The mask can have bits sticking out of it, if lines intersected with the crossword. We start
    # by median blurring to "cut" the lines.
    mask = cv2.medianBlur(mask, 51)[1:1+input.shape[0], 1:1+input.shape[1]]

    # But if some of the lines went into big, thick things, they'll still be part of the mask.
    # Therefore we fill out from the center point again.
    outputMask = np.zeros((filled.shape[0] + 2, filled.shape[1] + 2), np.uint8)
    cv2.floodFill(mask, outputMask, center, (255, 255, 255))

    outputMask = outputMask[1:1+input.shape[0], 1:1+input.shape[1]]
    
    show(outputMask*255)

    return np.uint8(outputMask*255)

# orthogonal truncated crossword
def get_cw_orth_trunc(input):
    mask = get_cw_mask(input)

    # get angle and rotate appropriately
    angle = get_angle_hough(mask)
    mask = rotate(mask, angle)
    input = rotate(input, angle)

    whites = cv2.findNonZero(mask)
    rect = cv2.boundingRect(whites)
    return input[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

# get grid row count
def get_grid_row_count(input):
    tmp = cv2.Canny(input, 50, 200)

    # get line spacings
    mx = input.shape[0] + 1
    real_mx = 0
    real_mn = 10000000
    vals = np.zeros(mx)

    first = True
    lines = []
    thresh = 0
    while first or len(lines) > 100:
        first = False
        thresh += 10
        lines = cv2.HoughLines(tmp, 5, math.pi/180, thresh, 0, 0)

    deb = input.copy()
    for thingo in lines:
        theta = thingo[0][1]
        rho = abs(thingo[0][0])
        # only take things that are within the image and vaguely horizontal
        if rho >= mx or abs(math.cos(theta)) >= 0.1:
            continue

        # Get the closest point on the line to the origin
        a = math.cos(theta)
        b = math.sin(theta)
        x0 = a * rho
        y0 = b * rho
        # The line is given by (x0, y0) + c * (-b, a), where c is real.
        # We want to find the point in the middle of the image horizontally.
        # x0 - c * b = cols / 2, so c = (x0 - cols / 2) / b
        ycenter = int(y0 + ((x0 - input.shape[1] / 2) / b) * a)
        if ycenter >= mx or ycenter < 0:
            continue
        vals[ycenter] += 1
        real_mx = max(real_mx, ycenter)
        real_mn = min(real_mn, ycenter)

        if debug:
            pt1 = (int(x0 + 1000*-b), int(y0 + 1000*a))
            pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*a))
            cv2.line(deb, pt1, pt2, (0, 0, 255), 1)

    show(deb)

    mags = np.absolute(np.fft.fft(vals[real_mn:real_mx+1]))
    thresh = np.max(mags) / 2
    # take the first peak after fst that's over the threshold
    fst = 9
    last = mags[fst]
    for i in range(fst + 1, len(mags)):
        ti = mags[i]
        if ti < last and last > thresh:
            return i - 1
        last = ti
    raise Exception("Failed to get grid count")

def is_black_square(input, row_count, col_count, row, col):
    spr = float(input.shape[0]) / float(row_count);
    spc = float(input.shape[1]) / float(col_count);
    # get actual row/col pixel
    r = int(float(row) * spr + spr / 2);
    c = int(float(col) * spc + spc / 2);

    dimr = int(spr/4)
    dimc = int(spc/4)
    left = max(0, c - dimc)
    top = max(0, r - dimr)
    width = min(input.shape[1] - left, 2 * dimc)
    height = min(input.shape[0] - top, 2 * dimr)
    masked = input[top:top+height, left:left+width]
    whites = np.nonzero(masked)
    return len(whites[0]) < width * height / 2

def get_grid(input):
    show(input)
    cw = get_cw_orth_trunc(input)

    cw = cw.copy()
    cw = do_threshold(cw)
    show(cw)

    width = get_grid_row_count(cv2.transpose(cw))
    height = get_grid_row_count(cw)

    black = [[is_black_square(cw, height, width, r, c) for c in range(width)] for r in range(height)]
    return (black, width, height)

def apply(input):
    if "b64data" in input and isinstance(input["b64data"], bytes):
        image_data_base64 = input["b64data"]
    else:
        raise Exception("b64data input missing or malformed")
    image = cv2.imdecode(np.fromstring(base64.b64decode(image_data_base64), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    black, width, height = get_grid(image)
    result = str(width) + " " + str(height)
    for row in black:
        result += "|"
        for col in row:
            if col:
                result += "#"
            else:
                result += " "
    return result

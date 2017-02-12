import Algorithmia
import base64
import cv2
import numpy as np
import math

# helper to convert an angle (in radians) to [-pi, pi)
def principal_angle(angle):
    tmp = math.fmod(angle, 2 * math.pi) # (-2pi, 2pi)
    if tmp < 0:
        tmp += 2 * math.pi
    # [0, 2pi)
    return math.fmod(tmp + math.pi, 2 * math.pi) - math.pi # [-pi, pi)

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
    (_,filled) = cv2.threshold(filled, 128., 255., cv2.THRESH_BINARY)
    # Fill from all corners
    ini = 1

    col = (0, 0, 0)
    mask = np.zeros((filled.shape[0] + 2, filled.shape[1] + 2), np.uint8)
    cv2.floodFill(filled, mask, (ini,ini), col)
    cv2.floodFill(filled, mask, (filled.shape[1] - ini,ini), col)
    cv2.floodFill(filled, mask, (filled.shape[1] - ini,filled.shape[0] - ini), col)
    cv2.floodFill(filled, mask, (ini,filled.shape[0] - ini), col)
    # Find average white pixel
    tr = 0
    tc = 0
    locs = np.nonzero(filled)
    nlocs = len(locs[0])
    for i in range(nlocs):
      tc += locs[1][i]
      tr += locs[0][i]

    oldmask = mask.copy()

    bc = (255, 255, 255)
    foo = cv2.floodFill(filled, mask, (int(float(tc) / float(nlocs)), int(float(tr) / float(nlocs))), (255, 0, 0), bc, bc)
    return np.nonzero(foo[1])
    mask -= oldmask
    outputMask = mask[1:1+input.shape[0], 1:1+input.shape[1]]
    
    #outputMask.convertTo(outputMask, CV_8UC1, 255.);
    return outputMask

# orthogonal truncated crossword
def get_cw_orth_trunc(input):
    mask = get_cw_mask(input)
    return mask

    # get angle and rotate appropriately
    angle = get_angle_hough(mask)
    mask = rotate(mask, angle)
    input = rotate(input, angle)

    whites = cv2.findNonZero(mask)
    rect = cv2.boundingRect(whites)
    return input[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]]

# get grid count (assumes square)
def get_grid_count(input):
    tmp = cv2.Canny(input, 50, 200)

    # get line spacings
    mx = max(input.shape)
    vals = np.zeros(mx)

    first = True
    lines = []
    thresh = 0
    while first or len(lines) > 100:
        first = False
        thresh += 10
        lines = cv2.HoughLines(tmp, 5, math.pi/180, thresh, 0, 0)
        
    for thingo in lines:
        theta = thingo[0][1]
        rho = abs(thingo[0][0])
        # only take things that are within the image and vaguely orthogonal
        if rho < mx and (abs(math.cos(theta)) < 0.1 or abs(math.sin(theta)) < 0.1):
            vals[int(rho)] = vals[int(rho)] + 1

    # TODO: This might be wrong, since I replaced the below old version
    """
    Mat planes[] = {Mat_<float>(vals), Mat::zeros(vals.size(), 1, CV_32F)};
    Mat complexI;
    merge(planes, 2, complexI);
    dft(complexI, complexI);
    split(complexI, planes);
    magnitude(planes[0], planes[1], planes[0]);
    Mat magI = planes[0];
    # get the 90th percentile
    vector<float> dems;
    for (int i = 0; i < magI.rows; ++i) {
      dems.push_back(magI.at<float>(i, 0));
    }
    sort(dems.begin(), dems.end());
    float accept_thresh = dems[dems.size() * 9 / 10];
    // take the first peak after fst that's over the 90th percentile
    int fst = 9;
    float last = magI.at<float>(fst, 0);
    for (int i = fst + 1; i < magI.rows; ++i) {
      float ti = magI.at<float>(i, 0);
      if (ti < last && last > accept_thresh) {
        return i - 1;
      }
      last = ti;
    }
    cerr << "Oh no didn't find a grid count";
    return 1;
    """
    # with this:
    mags = np.absolute(np.fft.fft(vals))
    thresh = np.percentile(mags, 90)
    # take the first peak after fst that's over the 90th percentile
    fst = 9
    last = mags[fst]
    for i in range(fst + 1, len(mags)):
        ti = mags[i]
        if ti < last and last > thresh:
            return i - 1
        last = ti
    # TODO: Raise some kind of error
    return 1

def is_black_square(input, grid_count, row, col):
    sp = float(input.shape[0]) / float(grid_count);
    # get actual row/col pixel
    r = int(float(row) * sp + sp / 2);
    c = int(float(col) * sp + sp / 2);

    (_, tmp) = cv2.threshold(input, 128., 255., cv2.THRESH_BINARY)
    dim = int(sp/4)
    left = max(0, c - dim)
    top = max(0, r - dim)
    width = min(tmp.shape[1] - left, 2 * dim)
    height = min(tmp.shape[0] - top, 2 * dim)
    masked = tmp[top:top+height, left:left+width]
    whites = np.nonzero(masked)
    return len(whites[0]) < width * height / 2

def get_grid(input):
    return ("hi", get_cw_orth_trunc(input))
    cw = get_cw_orth_trunc(input)

    width = get_grid_count(cw)

    black = [[is_black_square(cw, width, r, c) for c in range(width)] for r in range(width)]
    return (black, width)

def apply(input):
    if "b64data" in input and isinstance(input["b64data"], basestring):
        # TODO: Don't do this split.
        image_data_base64 = input["b64data"].split(",")[1]
    else:
        # TODO: Actually throw an error here
        image_data_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QCaRXhpZgAASUkqAAgAAAAEABIBAwABAAAAAQAAADEBAgAWAAAAPgAAADIBAgAUAAAAVAAAAGmHBAABAAAAaAAAAAAAAABDaHJvbWUgT1MgR2FsbGVyeSBBcHAAMjAxNzowMjowNCAyMDo1MToyMwADAACQBwAEAAAAMDIyMAKgBAABAAAAFAAAAAOgBAABAAAAEgAAAAAAAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAASABQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDj/hf490TwH4hkDRRP4etrQWl7ZaZcyh3idmSQRjeoZhI7OGZtxCgcclfUPEngHQtE8D62mly33jC28Rm1ljaO2SS3tTat8kM8iN5iP+8kwBGDtKqN4+c/NPhTQ9Vi8RWEl8kd7F9oJujaxvJw+6NhvdSOVkUgjyj8zZKhefpPVNG/4V54bsbN/E0N4ms3AOnCzKLam1jcDZICyyJJvkjYrtYDBKhdrs3NgYUZyhSxEuWOqcrN2T66b6Nq3lfdn2mcwxeGo1KmXpuo7OMU+Xm5U3a9vdcbKV3vd3+FW4HwX4hvfAPhTTNL1mw04XSxE51aayin+VjGwKSyIy7XR0wRxsxRWjeeFfDk6W7X+nRS7leSGVZnjDpJI8hYFSAwLu5z+HGMAr0H4f5jjG8Rg5RnSlrGSkrNPZr3tux+ST49yenJxx3tIVV8UZRqcyl1TsrXvuc3ffuLrSkj/dpNqhjkVeA6mOUlW9QcDg+grf0rULqDWfG2nR3M0enjwfd6gLRZCIhdRlnjn2dPMVgGD43AgEHNFFePP+LL1P3+r/uv3fmj6I+ErFfCsiKSFW8nCqDwPnJ4/Emiiiv4Jzv/AJGmJ/xy/wDSmepQ/hQ9F+R//9k="
    image = cv2.imdecode(np.fromstring(base64.b64decode(image_data_base64), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    black, width = get_grid(image)
    return width
    result = "Width " + str(width)
    for row in black:
        result += "\n"
        for col in row:
            if col:
                result += "#"
            else:
                result += " "
    return result
import Algorithmia
import base64
import cv2
import numpy as np

def apply(input):
    if "b64data" in input and isinstance(input["b64data"], basestring):
        image_data_base64 = input["b64data"]
    else:
        image_data_base64 = "/9j/4AAQSkZJRgABAQAAAQABAAD/4QCaRXhpZgAASUkqAAgAAAAEABIBAwABAAAAAQAAADEBAgAWAAAAPgAAADIBAgAUAAAAVAAAAGmHBAABAAAAaAAAAAAAAABDaHJvbWUgT1MgR2FsbGVyeSBBcHAAMjAxNzowMjowNCAyMDo1MToyMwADAACQBwAEAAAAMDIyMAKgBAABAAAAFAAAAAOgBAABAAAAEgAAAAAAAAD/2wBDAAMCAgMCAgMDAwMEAwMEBQgFBQQEBQoHBwYIDAoMDAsKCwsNDhIQDQ4RDgsLEBYQERMUFRUVDA8XGBYUGBIUFRT/2wBDAQMEBAUEBQkFBQkUDQsNFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBQUFBT/wAARCAASABQDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwDj/hf490TwH4hkDRRP4etrQWl7ZaZcyh3idmSQRjeoZhI7OGZtxCgcclfUPEngHQtE8D62mly33jC28Rm1ljaO2SS3tTat8kM8iN5iP+8kwBGDtKqN4+c/NPhTQ9Vi8RWEl8kd7F9oJujaxvJw+6NhvdSOVkUgjyj8zZKhefpPVNG/4V54bsbN/E0N4ms3AOnCzKLam1jcDZICyyJJvkjYrtYDBKhdrs3NgYUZyhSxEuWOqcrN2T66b6Nq3lfdn2mcwxeGo1KmXpuo7OMU+Xm5U3a9vdcbKV3vd3+FW4HwX4hvfAPhTTNL1mw04XSxE51aayin+VjGwKSyIy7XR0wRxsxRWjeeFfDk6W7X+nRS7leSGVZnjDpJI8hYFSAwLu5z+HGMAr0H4f5jjG8Rg5RnSlrGSkrNPZr3tux+ST49yenJxx3tIVV8UZRqcyl1TsrXvuc3ffuLrSkj/dpNqhjkVeA6mOUlW9QcDg+grf0rULqDWfG2nR3M0enjwfd6gLRZCIhdRlnjn2dPMVgGD43AgEHNFFePP+LL1P3+r/uv3fmj6I+ErFfCsiKSFW8nCqDwPnJ4/Emiiiv4Jzv/AJGmJ/xy/wDSmepQ/hQ9F+R//9k="
    image_data = base64.b64decode(image_data_base64)
    image = cv2.imdecode(np.fromstring(image_data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    return "Dimensions " + str(image.shape)
    #return "hello {}".format(input) + "opencv version: " + cv2.__version__

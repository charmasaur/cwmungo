#from . import crossword_extractor
import base64
import crossword_extractor
import os

def test_crossword_extractor():
    for i in range(get_num()):
        print("Testing " + str(i) + "...")
        test(i)
        print("  Success")
    print("All tests passed!")

def test(test_num):
    b64data = load_input(test_num)
    actual = crossword_extractor.apply({"b64data":b64data})
    expected = load_expected(test_num)
    assert_equal(expected, actual)

def get_num():
    num_files = len(os.listdir("test_data"))
    if num_files % 2 == 1:
        raise Exception("Odd number of files")
    return num_files / 2

def load_input(test_num):
    f = open("test_data/in" + str(test_num) + ".jpg", "rb")
    data = f.read()
    f.close()
    return base64.b64encode(data)

def load_expected(test_num):
    f = open("test_data/out" + str(test_num) + ".txt", "r")
    data = f.read()
    f.close()
    return data

def assert_equal(expected, actual):
    if not expected == actual:
        print("Expected/actual:")
        print(expected)
        print(actual)
        raise Exception("Failure")

test_crossword_extractor()

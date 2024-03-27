import sys

with open("output.txt", 'w') as f:
    original_stdout = sys.stdout

    sys.stdout = f

    print("hello world")

    sys.stdout = original_stdout
import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
print(sys.path)
print("\n")
sys.path.append(BASE_DIR)
print(sys.path)
print(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../"))
print(sys.path)
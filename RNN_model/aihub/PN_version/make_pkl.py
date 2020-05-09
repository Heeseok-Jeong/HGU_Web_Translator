import pickle
from collections import OrderedDict


key_file_name = "./vocab/extractPN.kr"
val_file_name = "./vocab/extractPN.en"
new_file_name = "./vocab/extractPN.pkl"
content_file_name = "./vocab/extractPN_content.txt"

# myDict = OrderedDict()
myDict = dict()

key_file = open(key_file_name, "r", encoding="utf-8")
val_file = open(val_file_name, "r", encoding="utf-8")

key_line = key_file.readline()
val_line = val_file.readline()
while key_line:
    key_line = key_line.replace('\n', '')
    val_line = val_line.replace('\n', '')
    key_line = key_line.replace('\xa0', '')
    myDict[key_line] = val_line
    key_line = key_file.readline()
    val_line = val_file.readline()

with open(new_file_name, "wb")as nf:
    pickle.dump(myDict, nf)

with open(new_file_name, "rb")as nf:
    print_dict = pickle.load(nf)
print(print_dict)
print(type(print_dict))
# print(print_dict[0])

with open(content_file_name, "w")as nf:
    nf.write(str(print_dict))


key_file.close()
val_file.close()

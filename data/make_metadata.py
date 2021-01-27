import os 
import random

f1 = open("ljspeech/metadata.txt","r")
f2 = open("numienbac/metadata.txt", "r")
train = open("train.txt","a")
test = open("val.txt","a")

data = []
id = 1
for line in f1.readlines():
    filename = "ljspeech/wavs/"+line.split("|")[0] + ".wav"
    language = "en-us"
    speaker = "0"
    text = line.split("|")[1]
    data.append(f"{id}|{speaker}|{language}|{filename}|{text}")
    id+=1

for line in f2.readlines():
    filename = "numienbac/wavs/"+line.split("|")[0] + ".wav"
    language = "vi"
    speaker = "1"
    text = line.split("|")[1]
    data.append(f"{id}|{speaker}|{language}|{filename}|{text}")
    id+=1

random.shuffle(data)
# print(data)
x = 1
for d in data:
    if x < 300:
        test.write(d)
    else:
        train.write(d)
    x += 1
f1.close()
f2.close()
test.close()
train.close()

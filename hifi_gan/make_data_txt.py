import glob 
import os 
# trains = []
# tests = []

# count = 0
# for path in glob.glob('wavs_raw/*.wav'):
#     if count <= 100:
#         tests.append(path.split('/')[-1].split('.')[0])
#     else:
#         trains.append(path.split('/')[-1].split('.')[0])
#     count+=1

# f = open('training.txt','a')
# g = open('validation.txt','a')

# for test in tests:
#     g.write(test+"\n")
# for train in trains:
#     f.write(train+"\n")

# f.close()
# g.close()

for path in glob.glob('data/vlsp/wavs/*.wav'):
    filename = path.split('/')[-1]
    os.system(f'sox {path} -r 22050 data/vlsp/wavs2/{filename}')
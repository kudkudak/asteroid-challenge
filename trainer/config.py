import os

SplitChannels = True

##### Constants #####
DataAugDir = "/data/data_aug_final_8"
ImageChannels = int(4)

if SplitChannels:
    ImageChannels = int(1)

ImageSide = int(64)
MaximumPixelIntensity = 256.0 # Converted to log scale was 65553

DataCache = "/data/data_caches" # RAM Memory HA :)

AllColumns = 64
ColumnsResult = [15,31,47,63]
ImportantColumns = [9, 10, 11, 12, 13, 14, 9+16, 10+16, 11+16, 12+16, 13+16, 14+16,\
                    9+32, 10+32, 11+32, 12+32, 13+32, 14+32,\
                    9+48, 10+48, 11+48, 12+48, 13+48, 14+48\
                    ]
ImportColumnCount = 24
ColumnsResultCoiunt = 4

ExtraColumns = ImportColumnCount

#### Simple calculations
rawdataset_files = [os.path.join("data", f) for f in next(os.walk("data"))[2] if f.endswith(".raw")]
rawdataset_size = len(rawdataset_files)

if SplitChannels:
    ImageChannels = int(1)
    ImportantColumns = [9,10,11,12,13,14]
    ImportColumnCount = 6
    ColumnsResultCoiunt = 1
    ColumnsResult = [15] 
    ExtraColumns=ImportColumnCount


os.system("mkdir "+DataAugDir)


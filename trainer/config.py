import os

##### Constants #####
DataAugDir = "data_aug_final_8_diff"
ImageChannels = int(4)
ImageSide = int(64)
ExtraColumns = 1+1+1+1+1+1 # Magnitu + FWHM+ Theta + Elong + RMSE + Deltamu
MaximumPixelIntensity = 65535.0

DataCache = "data_caches" # RAM Memory HA :)

#### Simple calculations
rawdataset_files = [os.path.join("data", f) for f in next(os.walk("data"))[2] if f.endswith(".raw")]
rawdataset_size = len(rawdataset_files)


os.system("mkdir "+DataAugDir)


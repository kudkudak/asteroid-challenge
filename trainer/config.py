import os

##### Constants #####
DataAugDir = "/data/data_aug_final_16"
ImageChannels = int(4)
ImageSide = int(64)
ExtraColumns = 1+1+1+1+1+1 # Magnitu + FWHM+ Theta + Elong + RMSE + Deltamu
MaximumPixelIntensity = 256.0 # Converted to log scale was 65553

DataCache = "/data/data_caches" # RAM Memory HA :)

#### Simple calculations
rawdataset_files = [os.path.join("data", f) for f in next(os.walk("data"))[2] if f.endswith(".raw")]
rawdataset_size = len(rawdataset_files)


os.system("mkdir "+DataAugDir)


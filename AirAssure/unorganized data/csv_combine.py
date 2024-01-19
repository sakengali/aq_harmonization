import pandas as pd
import os
import glob

folders = [name for name in os.listdir(".") if os.path.isdir(name)]
print(folders)

for folder in folders:
    os.chdir(folder)
    extension = 'csv'
    all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
    #combine all files in the list
    combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
    #export to csv
    combined_csv.to_csv( "../" + folder + ".csv", index=False, encoding='utf-8-sig')
    os.chdir("..")

import glob
import os
categories = ["train", "val", "test"]
for category in categories:
  subdirs = [os.path.basename(x) for x in glob.glob(f"./{category}/*")]
  for subdir in subdirs:
    file_names = [os.path.basename(x) for x in glob.glob(f"./{category}/{subdir}/*")]
    with open(f'./{category}/{subdir}.txt', 'w') as f:
      for file_name in file_names:
        f.write(file_name)
        f.write('\n')


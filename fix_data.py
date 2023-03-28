import os

# Cluster Root (root = "/home/accelerator/bands_test")

root = "/home/accelerator/bands_test"

for dirpath, dirnames, filenames in os.walk(root):
  for filename in filenames:
    if filename.endswith(".tif"):
      continue

    name, ext = os.path.splitest(filename)
    new_file = file_name.replaace(".", "_") + ext

    old_path = os.path.join(dirpath, filename)
    new_path = os.path.join(dirpath, new_file)

    os.rename(old_path, new_path)

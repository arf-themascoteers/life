import shutil
import os


def delete_all_files(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)


delete_all_files("saved_graphics")
delete_all_files("results")

import shutil
import os



def delete_all_files(folder_path):
    shutil.rmtree(folder_path)
    os.makedirs(folder_path)


def do_it():
    delete_all_files("saved_graphics")
    delete_all_files("results")


if __name__ == "__main__":
    do_it()


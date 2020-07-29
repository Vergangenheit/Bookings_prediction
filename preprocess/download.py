from config import config
import os
from google_drive_downloader import GoogleDriveDownloader as gd
import concurrent.futures
import time
import pandas as pd

# data_path = os.path.join(os.getcwd(), config.PATH)

files_dict = {config.ACTION_TRAIN_FILE: config.ACTIONS_FILE_ID,
              config.BOOKINGS_TRAIN_FILE: config.BOOKINGS_FILE_ID}
files = [config.ACTION_TRAIN_FILE, config.BOOKINGS_TRAIN_FILE]


def download(file):
    if not os.path.exists(config.PATH):
        os.makedirs(config.PATH)
    gd.download_file_from_google_drive(file_id=files_dict[file],
                                       dest_path=os.path.join(config.PATH, file),
                                       unzip=True)


def run():
    t1 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(download, files)
    #load actions file and count n unique actions
    actions = actions = pd.read_csv(os.path.join(config.PATH, config.ACTIONS_FILE_ID), delimiter='\t')
    no_unique_actions = actions['action_id'].nunique()
    t2 = time.perf_counter()
    print(f'Finished in {t2 - t1} seconds')

    return no_unique_actions


if __name__ == "__main__":
    run()

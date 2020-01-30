from config import config
import os
from google_drive_downloader import GoogleDriveDownloader as gd

# url = requests.get('https://drive.google.com/open?id=1paXv0q1YVZkMQcJQpi5PdCGLnmwtmW5D')
# csv_raw = StringIO(url.text)
# print(str(csv_raw)[0])
# actions = pd.read_csv(csv_raw, delimiter='\t')
# print(actions.head(10))

if not os.path.exists(config.PATH):
    os.makedirs(config.PATH)
gd.download_file_from_google_drive(file_id=config.ACTIONS_FILE_ID,
                                   dest_path=os.path.join(config.PATH, config.ACTION_TRAIN_FILE),
                                   unzip=True)
gd.download_file_from_google_drive(file_id=config.BOOKINGS_FILE_ID,
                                   dest_path=os.path.join(config.PATH, config.BOOKINGS_TRAIN_FILE),
                                   unzip=True)

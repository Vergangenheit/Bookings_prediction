from config import config
import os
import pandas as pd

actions = pd.read_csv(os.path.join(config.PATH, config.ACTION_TRAIN_FILE),
                      delimiter='\t')
print(actions.head())
print(actions.step.max())
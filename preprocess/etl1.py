from config import config
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
import pickle


def clean_sessions() -> list:
    actions = pd.read_csv(os.path.join(config.PATH, config.ACTION_TRAIN_FILE),
                          delimiter='\t')
    session_list = actions.session_id.unique().tolist()
    n_actions = []
    for session in tqdm(session_list):
        n_step = actions[actions['session_id'] == session].step.max()
        n_actions.append(n_step)

    cut_off = int(np.mean(n_actions) + np.std(n_actions))
    filt = np.array(n_actions) <= cut_off
    session_array = np.array(session_list)[filt]
    session_list_cleaned = session_array.tolist()

    return session_list_cleaned


def split_and_pivot(session_list: list):
    lower_limit = 0
    batch_size = int(len(session_list) / 10)
    upper_limit = batch_size

    for i in tqdm(range(10)):

        if upper_limit < len(session_list):

            actions = pd.read_csv(os.path.join(config.PATH, config.ACTION_TRAIN_FILE), delimiter='\t')
            actions = actions[actions['session_id'].isin(session_list[lower_limit:upper_limit])]
            actions = actions.pivot_table(index=['session_id'], columns=['step'], values=['action_id'])
            actions.columns = actions.columns.droplevel()
            actions = actions.reset_index()
            for u in range(actions.columns[-1] + 1, 60):
                actions[u] = np.nan
            actions.fillna(0, inplace=True)
            bookings = pd.read_csv(os.path.join(config.PATH, config.BOOKINGS_TRAIN_FILE),
                                   delimiter='\t', usecols=[2, 7])
            bookings = bookings[bookings['session_id'].isin(actions.session_id.tolist())]

            act_book = pd.merge(bookings, actions)

            if not os.path.exists(os.path.join(config.PATH, 'ActBook')):
                os.makedirs(os.path.join(config.PATH, 'ActBook'))
            act_book.to_csv(os.path.join(config.PATH, 'ActBook', 'act_book' + str(i) + '.csv'), index=True)

            lower_limit += batch_size
            upper_limit += batch_size


def merge(files_folder: str) -> pd.DataFrame:
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        return [atoi(c) for c in re.split(r'(\d+)', text)]

    files = os.listdir(os.path.join(config.PATH, files_folder))
    files.sort(key=natural_keys)

    sessions_merged = pd.DataFrame(data=None, columns=None)
    for file in files:
        sessions_merged = pd.concat([sessions_merged, pd.read_csv(os.path.join(config.PATH, files_folder, file),
                                                                  index_col=0)], axis=0, ignore_index=True)

    return sessions_merged

# sessions_merged = merge('ActBook')

from config import config
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

actions = pd.read_csv(os.path.join(config.PATH, config.ACTION_TRAIN_FILE),
                      delimiter='\t')
print(actions.head())
print(actions.step.max())


def clean_sessions() -> list:
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

            act_book.to_csv(os.path.join(config.PATH, 'act_book' + str(i) + '.csv'), index=True)

            lower_limit += batch_size
            upper_limit += batch_size

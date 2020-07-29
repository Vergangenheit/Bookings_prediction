from preprocess.download import run
from preprocess.etl1 import merge, clean_sessions, split_and_pivot, extract_feature_arrays
from model.model import Model

if __name__ == "__main__":
    #download
    no_unique_actions = run()

    #preprocess
    session_list_cleaned = clean_sessions()
    split_and_pivot(session_list_cleaned)
    sessions_merged = merge('ActBook')
    print(sessions_merged.head(10))
    X_train, X_val, y_train, y_val = extract_feature_arrays(sessions_merged)
    #train
    model = Model(no_unique_actions=no_unique_actions)
    model.build()
    model.train(X_train, X_val, y_train, y_val)



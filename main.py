from preprocess.download import run
from preprocess.etl1 import merge, clean_sessions, split_and_pivot, extract_feature_arrays

if __name__ == "__main__":
    #download
    run()

    #preprocess
    session_list_cleaned = clean_sessions()
    split_and_pivot(session_list_cleaned)
    sessions_merged = merge('ActBook')
    print(sessions_merged.head(10))
    #train


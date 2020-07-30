from preprocess.download import run
from preprocess.etl1 import merge, clean_sessions, split_and_pivot, extract_feature_arrays, extract_no_actions
from model.model import Model, cls_weights

if __name__ == "__main__":
    #download
    no_unique_actions = run()

    #preprocess
    session_list_cleaned = clean_sessions()
    split_and_pivot(session_list_cleaned)
    sessions_merged = merge('ActBook')
    print(sessions_merged.head(10))
    act_to_index = extract_no_actions(sessions_merged)
    class_weights = cls_weights(sessions_merged)
    X_train, X_val, y_train, y_val = extract_feature_arrays(sessions_merged)
    #train
    model = Model(no_unique_actions=len(act_to_index)+1)
    model.build()
    model.train(X_train, X_val, y_train, y_val, class_weights)
    model.save_model()



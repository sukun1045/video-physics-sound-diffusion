import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

if __name__ == "__main__":
    data_root = '/data/great_hits/'
    latent_root = data_root + 'hits_ten_mat_spec_diff_params_latent_video_fea_latents/train.pickle'
    train_video_root = data_root + 'hits_ten_mat_video_classifer_all_features/train.pickle'
    test_video_root = data_root + 'hits_ten_mat_video_classifer_all_features/test.pickle'

    with open(latent_root, 'rb') as handle:
        train_latent_data = pickle.load(handle)
    with open(train_video_root, 'rb') as handle:
        train_video_features = pickle.load(handle)
    with open(test_video_root, 'rb') as handle:
        test_video_features = pickle.load(handle)
    train_keys = list(train_video_features.keys())
    test_keys = list(test_video_features.keys())

    save_dict = {}
    for test_k in test_keys:
        test_v = test_video_features[test_k] #22, 2048
        avg_test_v = np.mean(test_v, axis=0)
        print(avg_test_v.shape)
        query_k = {'fn':None, 'dist':1e6}
        for train_k in train_keys:
            train_v = train_video_features[train_k]
            avg_train_v = np.mean(train_v, axis=0)
            dist = np.sum(abs(avg_train_v - avg_test_v))
            if query_k['fn'] is None or dist < query_k['dist']:
                query_k['fn'] = train_k
                query_k['dist'] = dist
        print(query_k)
        query_latent = train_latent_data[query_k['fn']]
        save_dict[test_k] = query_latent

    save_test_latent = data_root + 'hits_ten_mat_spec_diff_params_latent_video_fea_latents'

    with open(f'{save_test_latent}/query_test.pickle', 'wb') as handle:
        pickle.dump(save_dict, handle)


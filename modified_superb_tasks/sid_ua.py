import torch
import pandas as pd
import os
from s3prl.problem.common import SuperbSID

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class UASpeechSuperbSID(SuperbSID):
    def prepare_data(
        self, prepare_data: dict, target_dir: str, cache_dir: str, get_path_only=False
    ):
        #train_path, valid_path, test_paths = super().prepare_data(
        #    prepare_data, target_dir, cache_dir, get_path_only
        #)

        train_path = "/project/thesis/datasets/train_sid.csv"
        valid_path = "/project/thesis/datasets/test_sid.csv"
        test_paths = ["/project/thesis/datasets/test_sid.csv"]

        return train_path, valid_path, test_paths




if __name__ == "__main__":
    UASpeechSuperbSID().main()
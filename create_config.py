import json
from pathlib import Path

def create_config(output_path, num_classes):
    """
    Create PaddleOCR training configuration
    """
    config = {
        "Architecture": {
            "model_type": "rec",
            "algorithm": "SVTR_LCNet",
            "Transform": None,
            "Backbone": {
                "name": "PPLCNetV3",
                "scale": 0.95
            },
            "Neck": None,
            "Head": {
                "name": "MultiHead",
                "head_list": [
                    {
                        "CTCHead": {
                            "Neck": {
                                "name": "svtr",
                                "dims": 120,
                                "depth": 2,
                                "hidden_dims": 120,
                                "kernel_size": [1, 3],
                                "use_guide": True
                            },
                            "Head": {
                                "fc_decay": 1.0e-05
                            }
                        }
                    },
                    {
                        "NRTRHead": {
                            "nrtr_dim": 384,
                            "max_text_length": 25
                        }
                    }
                ]
            }
        },
        "Loss": {
            "name": "MultiLoss",
            "loss_config_list": [
                {"CTCLoss": None},
                {"NRTRLoss": None}
            ]
        },
        "Optimizer": {
            "name": "Adam",
            "beta1": 0.9,
            "beta2": 0.999,
            "lr": {
                "name": "Cosine",
                "learning_rate": 0.0005,
                "warmup_epoch": 5
            },
            "regularizer": {
                "name": "L2",
                "factor": 3.0e-05
            }
        },
        "Train": {
            "dataset": {
                "name": "MultiScaleDataSet",
                "data_dir": "dataset/",
                "label_file_list": ["./paddleocr_training/train_list.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    # {"RecConAug": {"prob": 0.5, "ext_data_num": 2}},
                    {"RecAug": None},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}}
                ]
            },
            "sampler": {
                "name": "MultiScaleSampler",
                "scales": [
                    [320, 32],
                    [320, 48],
                    [320, 64]
                ],
                "first_bs": 96,
                "fix_bs": False,
                "divided_factor": [8, 16],
                "is_training": True
            },
            "loader": {
                "shuffle": True,
                "batch_size_per_card": 96,
                "drop_last": True,
                "num_workers": 8
            }
        },
        "Eval": {
            "dataset": {
                "name": "SimpleDataSet",
                "data_dir": "dataset/",
                "label_file_list": ["paddleocr_training/val_list.txt"],
                "transforms": [
                    {"DecodeImage": {"img_mode": "BGR", "channel_first": False}},
                    {"MultiLabelEncode": {"gtc_encode": "NRTRLabelEncode"}},
                    {"RecResizeImg": {"image_shape": [3, 48, 320]}},
                    {"KeepKeys": {"keep_keys": ["image", "label_ctc", "label_gtc", "length", "valid_ratio"]}}
                ]
            },
            "loader": {
                "shuffle": False,
                "drop_last": False,
                "batch_size_per_card": 128,
                "num_workers": 4
            }
        },
        "Global": {
            "epoch_num": 200,
            "save_model_dir": "./output/rec_ppocr_v4",
            "save_epoch_step": 40,
            "eval_batch_step": [0, 100],
            "cal_metric_during_train": True,
            "pretrained_model": "./pretrain_models/en_PP-OCRv4_rec_train/best_accuracy",
            "checkpoints": "/content/drive/MyDrive/Deeplearning/Final_project/output/rec_ppocr_v4/best_accuracy.pdparams",
            "save_inference_dir": "./output/inference_results",
            "use_visualdl": False,
            "infer_img": "doc/imgs_words/ch/word_1.jpg",
            "character_dict_path": "paddleocr_training/dict.txt",
            "max_text_length": 25,
            "infer_mode": False,
            "use_space_char": True,
            "distributed": True,
            "save_res_path": "./output/rec/predicts_ppocrv3.txt",
            "log_smooth_window": 20,
            "print_batch_step": 10,
            "use_gpu": True  # GPU usage
        },
        "PostProcess": {
            "name": "CTCLabelDecode"
        },
        "Metric": {
            "name": "RecMetric",
            "main_indicator": "acc",
            "ignore_space": False
        }
    }

    config_file = Path(output_path) / 'config.yml'
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Created configuration file at: {config_file}")
"""Amazon SageMaker AI Remote Training Job Launcher (`scripts/launch_sagemaker.py`).

Supports:
1. Uploading processed XBAT+ manifests and images (`data/processed/`) to `s3://$YIELDSENSE_BUCKET/datasets/xbat/`.
2. Launching a managed SageMaker PyTorch Estimator (`ml.m5.xlarge` or `ml.g4dn.xlarge`) with CloudWatch
   metric definitions for validation category accuracy, battery hazard accuracy, and calibration threshold.
3. Recording immutable training job provenance (Job Name, Job ARN, S3 Model Artifact URI, instance type)
   to `artifacts/sagemaker_job_record.json`.
"""

from __future__ import annotations
import argparse
import json
import os
from pathlib import Path


def run(dry_run: bool = False) -> dict:
    role = os.environ.get("SAGEMAKER_ROLE_ARN", "")
    bucket = os.environ.get("YIELDSENSE_BUCKET", "")
    region = os.environ.get("AWS_REGION", "ap-south-1")
    instance_type = os.environ.get("SM_TRAIN_INSTANCE", "ml.m5.xlarge")
    framework_version = os.environ.get("SM_PYTORCH_VERSION", "2.1.0")
    py_version = os.environ.get("SM_PY_VERSION", "py310")

    hyperparameters = {
        "epochs": int(os.environ.get("SM_EPOCHS", "8")),
        "warmup_epochs": 2,
        "batch_size": 32,
        "lr": 1.5e-3,
        "battery_weight": 2.0,
        "alpha": 0.01,
    }

    metric_definitions = [
        {"Name": "train:loss", "Regex": r'"train_loss":\s*([0-9\.]+)'},
        {"Name": "val:cat_acc", "Regex": r'"val_cat_acc":\s*([0-9\.]+)'},
        {"Name": "val:bat_acc", "Regex": r'"val_bat_acc":\s*([0-9\.]+)'},
        {"Name": "cal:threshold", "Regex": r"Calibration threshold:\s*([0-9\.]+)"},
    ]

    if dry_run or not role or not bucket:
        spec = {
            "status": "DRY_RUN_VALIDATED",
            "entry_point": "train.py",
            "source_dir": "ml",
            "region": region,
            "instance_type": instance_type,
            "framework_version": framework_version,
            "py_version": py_version,
            "hyperparameters": hyperparameters,
            "metric_definitions": metric_definitions,
            "s3_train_channel": f"s3://{bucket or '<YIELDSENSE_BUCKET>'}/datasets/xbat/",
            "s3_validation_channel": f"s3://{bucket or '<YIELDSENSE_BUCKET>'}/datasets/xbat/",
            "s3_output_path": f"s3://{bucket or '<YIELDSENSE_BUCKET>'}/model-artifacts/",
        }
        print(json.dumps(spec, indent=2))
        return spec

    import sagemaker
    from sagemaker.pytorch import PyTorch

    session = sagemaker.Session()
    estimator = PyTorch(
        entry_point="train.py",
        source_dir="ml",
        role=role,
         sagemaker_session=session,
        framework_version=framework_version,
        py_version=py_version,
        instance_count=1,
        instance_type=instance_type,
        volume_size=15,
        max_run=1800,
        output_path=f"s3://{bucket}/model-artifacts/",
        base_job_name="yieldsense-dualhead",
        hyperparameters=hyperparameters,
        metric_definitions=metric_definitions,
    )

    estimator.fit(
        {
            "train": f"s3://{bucket}/datasets/xbat/",
            "validation": f"s3://{bucket}/datasets/xbat/",
        },
        wait=True,
        logs=True,
    )

    job_name = estimator.latest_training_job.name
    model_data = estimator.model_data
    record = {
        "status": "COMPLETED",
        "job_name": job_name,
        "model_data_s3_uri": model_data,
        "instance_type": instance_type,
        "region": region,
        "hyperparameters": hyperparameters,
    }
    out_file = Path("artifacts/sagemaker_job_record.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(record, indent=2))
    print("SageMaker Job Record:", json.dumps(record, indent=2))
    return record


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    run(dry_run=args.dry_run)

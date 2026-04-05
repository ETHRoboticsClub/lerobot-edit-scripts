from lerobot.datasets.lerobot_dataset import LeRobotDataset

ds = LeRobotDataset("ETHRC/towelspring26_realsense")

ds.finalize()  # Closes parquet writers, writes metadata footers
ds.push_to_hub()
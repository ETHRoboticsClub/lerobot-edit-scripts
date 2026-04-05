from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

# Load your 50-episode datasets
ds1 = LeRobotDataset("ETHRC/towelspring26_2")
ds2 = LeRobotDataset("ETHRC/towelspring26_3")

# Merge them into a 100-episode dataset
merged_dataset = merge_datasets(
    [ds1, ds2], 
    output_repo_id="ETHRC/towelspring26_realsense"
)

print(f"Merged dataset has {merged_dataset.meta.total_episodes} episodes!")
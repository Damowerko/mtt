from mtt.data import OnlineDataset

online_dataset = OnlineDataset(
    n_steps=119, sigma_position=10, length=20, img_size=128, device="cuda"
)

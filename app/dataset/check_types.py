from datasets import get_dataset_config_names

configs = get_dataset_config_names("EleutherAI/hendrycks_math")

print("\nAvailable Configurations:\n")
for c in configs:
    print(c)
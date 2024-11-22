import pandas as pd
import requests

df = pd.read_parquet("hf://datasets/huggan/smithsonian_butterflies_subset/data/train-00000-of-00001.parquet")

for i, r in df.iterrows():
    url = r["image_url"]
    response = requests.get(url)
    with open(f"images/{r['image_hash']}.jpg", "wb") as f:
        f.write(response.content)
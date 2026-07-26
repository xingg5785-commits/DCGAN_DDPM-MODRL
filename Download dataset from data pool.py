import os
from itertools import groupby
import matplotlib.pyplot as plt
from deep_translator import GoogleTranslator

# Setup environmental variables
os.environ["HF_TOKEN"] = "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"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# Setup download path
f_drive = "F:/hf_cache"
tempdir = "F:/hf_tmp"

os.environ["HF_HOME"] = f_drive
os.environ["TMP"] = tempdir
os.environ["TEMP"] = tempdir

import time
import datetime
import random
import shutil
import io
import zlib
import ast
import tempfile
import traceback
import numpy as np
import pandas as pd
import multiprocessing
import matplotlib.colors as mcolors
from tqdm import tqdm
from huggingface_hub import snapshot_download

# =====================================
# ==== Stable downloading function ====
# =====================================
def download_to_local(repo_id, local_name):
    target_path = os.path.join(f_drive, local_name)
    print(f"Downloading {repo_id} to local: {target_path}...")
    
    # Download Parquet files
    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        local_dir=target_path,
        max_workers=1,
        allow_patterns=["data/train-0000[0-9]-*"]
    )
    return target_path

# Filtering sub-datasets
def fast_sub_process(ds, cols, join_col, id_set):
    print(f"Processing dataset: {ds}")
    actual_cols = list(set(cols + [join_col]))
    df =ds.select_columns(actual_cols).to_pandas()
    print(f"Filtering {join_col}...")
    return df[df[join_col].isin(id_set)]

# ===========================
# ==== Decode level data ====
# ===========================
# Preview level data
def preview_levels(df, num_to_preview=5):
    print(f"Attempting to preview {num_to_preview} levels...")
    if 'level_data' not in df.columns:
        print("Error: 'level_data' column missing!")
        return
    
    valid_df = df[df['level_data'].notna()]
    if valid_df.empty:
        print("No valid level data found.")
        return
    
    valid_df = df[df['level_data'].notna()]
    sample = valid_df.sample(min(num_to_preview, len(valid_df)))
    
    # Visualization
    fig, axes = plt.subplots(num_to_preview, 1, figsize=(15, 3 * num_to_preview))
    if num_to_preview == 1: axes = [axes]
    
    count = 0
    max_attempt = 500
    attempt =0
    internal_h, grid_w = 32, 200
    actual_display_h = 27
    needed_bytes = internal_h * grid_w
    preview_width = 80
    
    while count < num_to_preview and attempt < max_attempt:
        attempt += 1
        row = valid_df.sample(1).iloc[0]
        
        try:
            bin_data = row['level_data']
            if isinstance(bin_data, str) and bin_data.startswith("b'"):
                bin_data = ast.literal_eval(bin_data)
            
            raw_data = zlib.decompress(bin_data)
            
            start_pos = 0x200
            end_pos = start_pos + needed_bytes
            map_bytes = np.frombuffer(raw_data[start_pos:end_pos], dtype=np.uint8).copy()
            
            if len(map_bytes) < needed_bytes:
                map_bytes = np.pad(map_bytes, (0, needed_bytes - len(map_bytes)))
            
            # Semantic mapping
            clean_grid = np.zeros_like(map_bytes)
            # Air 0 (blue)
            clean_grid[(map_bytes == 0)] = 0
            # Ground (brown)
            clean_grid[((map_bytes >= 1) & (map_bytes <= 2)) | (map_bytes >= 60)] = 1
            # Building block (gray)
            clean_grid[(map_bytes >= 3) & (map_bytes <= 15)] = 2
            # Gold coin/Props (yellow)
            clean_grid[(map_bytes >= 16) & (map_bytes <= 40)] = 3
            # Enemies (red)
            clean_grid[(map_bytes >= 41) & (map_bytes <= 59)] = 4
            
            full_grid = clean_grid.reshape((32, 200), order='F')
            
            img = full_grid[:27, :]
            
            if np.sum(img > 0) <50:
                continue
            
            # Set professional color disk
            custom_colors = ['#87CEEB', '#8B4513', '#808080', '#FFD700', '#FF0000']
            my_cmap = mcolors.ListedColormap(custom_colors)
            
            img_slice = img[:, :preview_width]
            
            axes[count].imshow(img_slice, aspect='equal', cmap=my_cmap, interpolation='nearest')
            
            axes[count].set_xticks(np.arange(-.5, preview_width, 1), minor=True)
            axes[count].set_yticks(np.arange(-.5, actual_display_h, 1), minor=True)
            axes[count].grid(which='minor', color='white', linestyle='-', linewidth=0.5, alpha=0.1)
            
            axes[count].set_xticks([])
            axes[count].set_yticks([])
            axes[count].set_title(f"Level ID: {row.get('data_id', 'Sample')} (Preview)")
            
            count += 1
            
        except Exception as e:
            traceback.print_exc()
            continue

    plt.tight_layout()
    plt.show()

# Compress all level data
def compress_level_data(df, save_path="D:/OneDrive/桌面/MM2_level_data.npz"):
    print(f"Starting compression for {len(df)} levels...")
    all_levels = []
    ids = []
    
    internal_h, grid_w = 32, 200
    actual_display_h = 27
    needed_bytes = internal_h * grid_w
    start_pos = 0x200
    end_pos = start_pos + needed_bytes
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            bin_data = row['level_data']
            if isinstance(bin_data, str) and bin_data.startswith("b'"):
                bin_data = ast.literal_eval(bin_data)
            
            raw_data = zlib.decompress(bin_data)
            map_bytes = np.frombuffer(raw_data[start_pos:end_pos], dtype=np.uint8).copy()
            
            # Semantic mapping
            clean_grid = np.zeros_like(map_bytes)
            # Air 0 (blue)
            clean_grid[(map_bytes == 0)] = 0
            # Ground (brown)
            clean_grid[((map_bytes >= 1) & (map_bytes <= 2)) | (map_bytes >= 60)] = 1
            # Building block (gray)
            clean_grid[(map_bytes >= 3) & (map_bytes <= 15)] = 2
            # Gold coin/Props (yellow)
            clean_grid[(map_bytes >= 16) & (map_bytes <= 40)] = 3
            # Enemies (red)
            clean_grid[(map_bytes >= 41) & (map_bytes <= 59)] = 4
            
            full_grid = clean_grid.reshape((32, 200), order='F')[:27:, :]
            
            if np.sum(full_grid > 0) <20:
                continue
            
            all_levels.append(full_grid.astype(np.uint8))
            ids.append(row.get('data_id', 'Unknown'))
            
        except Exception:
            continue
    
    if all_levels:
        np.savez_compressed(save_path, levels=np.array(all_levels), data_ids=np.array(ids))
        
        print(f"\nSucceed! {len(all_levels)} valid levels has been compressed to {save_path}.")
    else:
        print(f"\nError! no valid levels were found for compression.")

if __name__ == "__main__":
    from datasets import load_dataset
    
    os.makedirs(f_drive, exist_ok=True)
    os.makedirs(tempdir, exist_ok=True)
    
    level_path = download_to_local("TheGreatRambler/mm2_level", "mm2_level")
    comments_path = download_to_local("TheGreatRambler/mm2_level_comments", "mm2_level_comments")
    played_path = download_to_local("TheGreatRambler/mm2_level_played", "mm2_level_played")
    deaths_path = download_to_local("TheGreatRambler/mm2_level_deaths", "mm2_level_deaths")
    
    # ==== Load datasets from local ====
    print("Loading dataset from local drive...")
    level_ds = load_dataset("parquet", data_files=f"{level_path}/data/*.parquet", split="train")
    comments_ds = load_dataset("parquet", data_files=f"{comments_path}/data/*.parquet", split="train")
    played_ds = load_dataset("parquet", data_files=f"{played_path}/data/*.parquet", split="train")
    deaths_ds = load_dataset("parquet", data_files=f"{deaths_path}/data/*.parquet", split="train")
    
# ================================
# ==== Sample and merge logic ====
# ================================
    sample_size = 200000
    random.seed(42)
    
    print(f"Sampling {sample_size} samples from Hugging Face...")
    
    level_indices = random.sample(range(len(level_ds)), sample_size)
    sampled_levels = level_ds.select(level_indices)
    level_df = sampled_levels.to_pandas()
    
    target_ids = set(level_df["data_id"].unique())
    
    comments_df = fast_sub_process(comments_ds, ['data_id', 'type', 'text', 'has_beaten'], 'data_id', target_ids)
    comments_sub = comments_df.groupby(["data_id"]).agg({
        'type': 'first',
        'text': lambda x: " | ".join([str(i) for i in x if str(i).strip()]),
        'has_beaten':'first'
    }).reset_index()
    
    played_df = fast_sub_process(played_ds, ['data_id', 'cleared', 'liked'], 'data_id', target_ids)
    played_sub = played_df.groupby(["data_id"]).agg({
        'cleared': 'first',
        'liked': 'first'
    }).reset_index()
    
    deaths_df = fast_sub_process(deaths_ds, ['data_id', 'is_subworld'], 'data_id', target_ids)
    deaths_sub = deaths_df.groupby(["data_id"]).agg({
        'is_subworld': 'first'
    }).reset_index()
    
    print("Filtering sub-datasets...")
    
    # Merging for final aggregation
    print("Merging for all datasets...")
    merged_df = (level_df.merge(comments_sub, on="data_id", how="left") \
                         .merge(played_sub, on="data_id", how="left") \
                         .merge(deaths_sub, on="data_id", how="left"))
    
    print("merged_df columns:", merged_df.columns.tolist())
    
    # Preview level data
    preview_levels(merged_df, num_to_preview=5)
    
    # Compress all level data
    compress_level_data(merged_df)
    
    if not merged_df.empty:
        full_rules = {
        # ==== Primary table ====
        'name': 'first','gamestyle': 'first', 'theme': 'first', 'created': 'first', 'uploaded': 'first',
        'difficulty': 'first', 'timer': 'first', 'world_record': 'first', 'upload_attempts': 'first',
        'plays': 'first', 'likes': 'first', 'boos': 'first', 'tag1': 'first', 'tag2': 'first', 'clears': 'first',
        
        # ==== Subtable ====
        # played
        'cleared': 'first','liked': 'first',
        
        # deaths
        'is_subworld': 'first',
        
        # comments
        'has_beaten': 'first', 'type': 'first'
        }
        
        agg_rule = {k: v for k, v in full_rules.items() if k in merged_df.columns}
        
        merged_df = merged_df.groupby('data_id').agg(agg_rule).reset_index()
        
        merged_df['uploaded'] = pd.to_datetime(merged_df['uploaded'], unit='s', errors='coerce')
        merged_df['created'] = pd.to_datetime(merged_df['created'], unit='s', errors='coerce')
        
        # Rename columns
        merged_df.rename(
            columns={'name': 'level_name', 'created': 'created_timestamp', 'uploaded': 'uploaded_timestamp',
                     'timer': 'level_timer'}, inplace=True)
        
        output_path = "D:/OneDrive/桌面/Mario Maker Levels.csv"
        merged_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"Success saved {len(merged_df)} to {output_path}")
    else:
        print(f"Error: No matching comments found for the sample levels")
        print()
        
'''
        # Define dictionaries
        gamestyle_map = {
            0: "SMB1",
            1: "SMB3",
            2: "SMW",
            3: "NSMBU",
            4: "SM3DW"
        }
        
        difficulty_map = {
            0: "Easy",
            1: "Normal",
            2: "Expert",
            3: "Super expert"
        }
        
        theme_map = {
            0: "Overworld",
            1: "Underground",
            2: "Castle",
            3: "Airship",
            4: "Underwater",
            5: "Ghost house",
            6: "Snow",
            7: "Desert",
            8: "Sky",
            9: "Forest"
        }
        
        tag_map = {
            0: "None",
            1: "Standard",
            2: "Puzzle solving",
            3: "Speedrun",
            4: "Autoscroll",
            5: "Auto mario",
            6: "Short and sweet",
            7: "Multiplayer versus",
            8: "Themed",
            9: "Music",
            10: "Art",
            11: "Technical",
            12: "Shooter",
            13: "Boss battle",
            14: "Single player",
            15: "Link"
        }
        
        type_map = {
            0: "Custom Image",
            1: "Text",
            2: "Reaction Image"
        }
        
        common_map ={
            0: "No",
            1: "Yes"
        }
        
        # Mapping dictionaries for human understanding
        merged_df['gamestyle'] = merged_df['gamestyle'].map(gamestyle_map)
        merged_df['difficulty'] = merged_df['difficulty'].map(difficulty_map)
        merged_df['theme'] = merged_df['theme'].map(theme_map)
        merged_df['tag1'] = merged_df['tag1'].map(tag_map)
        merged_df['tag2'] = merged_df['tag2'].map(tag_map)
        merged_df['type'] = merged_df['type'].map(type_map)
        merged_df['cleared'] = merged_df['cleared'].map(common_map)
        merged_df['liked'] = merged_df['liked'].map(common_map)
        merged_df['has_beaten'] = merged_df['has_beaten'].map(common_map)
        merged_df['is_subworld'] = merged_df['is_subworld'].map(common_map)
'''

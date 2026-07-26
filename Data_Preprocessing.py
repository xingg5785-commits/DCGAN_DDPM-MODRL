import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.preprocessing import MinMaxScaler

# ========================================
# ======== Upload Level Thumbnail ========
# ========================================
def plot_level_thumbnails(npz_path, num=16, cols=4, start_idx=0):
	level_data = np.load(npz_path)
	levels = level_data['levels']
	data_ids = level_data['data_ids']
	
	# select level to be displayed
	indices = range(start_idx, start_idx + num)
	rows = (num + cols - 1) // cols
	
	# Use discrete color maps to distinguish different tiles
	custom_colors = ['#87CEEB', '#8B4513', '#808080', '#FFD700', '#FF0000']
	cmap = mcolors.ListedColormap(custom_colors)
	
	fig, axes = plt.subplots(rows, cols + 1, figsize=((cols + 1) * 5, rows * 2.5), constrained_layout=True)
	plot_axes = axes[:, :cols].flatten()
	cbar_axes = axes[:, cols]
	
	for i, idx in enumerate(indices):
		ax = plot_axes[i]
		im = ax.imshow(
			levels[idx],
			cmap=cmap,
			interpolation='nearest',
			aspect='auto',  # Adapt to an aspect ratio of 27x200
			vmin=0,
			vmax=4
		)
		ax.set_title(f"ID: {data_ids[idx]}", fontsize=7)
		ax.axis('off')
	
	# Hide redundant grids
	for j in range(num, len(axes)):
		axes[j].plot_axis('off')
	
	# Add a colorbar to display the tile ID comparison
	for ax in cbar_axes:
		ax.axis('off')
	cbar = fig.colorbar(im, ax=cbar_axes[:num].tolist(), label='Tile ID', shrink=0.6, pad=0.02)
	cbar.set_ticks([0, 1, 2, 3, 4])
	cbar.set_ticklabels(['Air', 'Ground', 'Block', 'Coin/Item', 'Enemy'])
	
	plt.suptitle(f"MM2 Level Thumbnails (index {start_idx}~{start_idx + num - 1})", fontsize=13, y=1.01)
	plt.show()

def get_processed_data(csv_path, npz_path):
	# ================================
	# ======== Upload Dataset ========
	# ================================
	merged_df = pd.read_csv(csv_path)
	# Explore Data Analysis
	print(f"The basic information of the merged dataset:")
	print(merged_df.info())
	print()
	
	print(f"The basic description of the merged dataset:")
	print(merged_df.describe())
	print()
	
	# Identify missing values
	print(f"The number of missing values in the merged dataset:")
	print(merged_df.isnull().sum())
	print()
	
	# Handling missing values
	merged_df.dropna(subset=['level_name'], inplace = True)
	merged_df = merged_df.reset_index(drop=True).copy()
	
	merged_df['tag1'] = merged_df['tag1'].fillna(0)
	merged_df['tag2'] = merged_df['tag2'].fillna(0)
	
	is_subworld_mode = merged_df['is_subworld'].mode()[0]
	merged_df['is_subworld'] = merged_df['is_subworld'].fillna(is_subworld_mode)
	
	has_beaten_mode = merged_df['has_beaten'].mode()[0]
	merged_df['has_beaten'] = merged_df['has_beaten'].fillna(has_beaten_mode)
	
	type_mode = merged_df['type'].mode()[0]
	merged_df['type'] = merged_df['type'].fillna(type_mode)
	
	# Verifying missing values
	print(f"The number of missing values in the merged dataset: {merged_df.isnull().values.sum()}")
	print()
	
	# Adding playability indicators
	def add_playability_metrics(df):
		metrics = {}
		
		metrics['satisfaction_rate'] = df['likes'] / (df['likes'] + df['boos'] + 1)
		
		like_rate = df['likes'] / (df['plays'] + 1)
		metrics['like_rate_norm'] = (like_rate - like_rate.min()) / (like_rate.max() - like_rate.min() + 1e-8)
		
		metrics['difficulty_score'] = np.exp(-((df['clears'] / (df['plays'] + 1)- 0.20) ** 2) / (2 * (0.15 ** 2)))
		
		metrics['playability_index'] = ((metrics['satisfaction_rate'] * 0.50) +
										  (metrics['like_rate_norm'] * 0.30) +
										  (metrics['difficulty_score'] * 0.20)
										  )
		
		metrics_df = pd.DataFrame(metrics, index=df.index)
		return pd.concat([df, metrics_df], axis=1)
	
	merged_df = add_playability_metrics(merged_df)
	
	print("Print out basic information of the merged dataset after adding playability metrics:")
	print(merged_df.info())
	print()
	
	# Upload and align NPZ data
	level_data = np.load(npz_path)
	levels = level_data['levels']
	data_ids = level_data['data_ids']
	
	npz_df = pd.DataFrame({'data_id': data_ids, 'level_index': range(len(data_ids))})
	
	# Make sure the type of data id is consistent
	merged_df['data_id'] = merged_df['data_id'].astype(str)
	npz_df['data_id'] = npz_df['data_id'].astype(str)
	
	# Fusion alignment
	if 'level_index' in merged_df.columns:
		merged_df = merged_df.drop(columns=['level_index'])
	
	aligned_df = pd.merge(merged_df, npz_df, on='data_id', how='inner')
	
	keep_indices = aligned_df['level_index'].values
	
	keep_indices = aligned_df['level_index'].values
	aligned_levels = levels[keep_indices]
	
	print(f"Original Levels: {len(levels)} -> Aligned Levels: {len(aligned_levels)}")
	print()
	print(f"Are there NaNs in levels? {np.isnan(aligned_levels).sum()}")
	print()
	
	# Feature scaling
	cols_to_scale = [
		'level_timer', 'world_record', 'upload_attempts', 'plays',
		'likes', 'boos', 'clears', 'clear_rate', 'satisfaction_rate',
		'like_rate', 'log_attempts', 'playability_index'
	]
	
	cols_to_scale = [col for col in cols_to_scale if col in aligned_df.columns]
	
	data_to_scale = aligned_df[cols_to_scale].fillna(0)
	
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_data = scaler.fit_transform(aligned_df[cols_to_scale].fillna(0))
	
	scaled_df = pd.DataFrame(scaled_data, columns=data_to_scale.columns, index=aligned_df.index)
	
	non_scaled_cols = ['data_id', 'gamestyle', 'theme', 'type', 'is_subworld', 'has_beaten']
	existing_non_scaled = [col for col in non_scaled_cols if col in aligned_df.columns]
	
	final_scaled_df = pd.concat([aligned_df[existing_non_scaled], scaled_df], axis=1)
	
	return aligned_df, final_scaled_df, aligned_levels, scaler
	
if __name__ == "__main__":
	csv_path = r"D:\OneDrive\桌面\Capstone Project\Mario Maker Levels.csv"
	npz_path = r"D:\OneDrive\桌面\Capstone Project\MM2_level_data.npz"
	
	plot_level_thumbnails(npz_path, num=16, cols=4, start_idx=0)
	
	aligned_df, final_scaled_df, aligned_levels, scaler = get_processed_data(csv_path, npz_path)
	print("Data preprocessing is completed. Scaled shape:", final_scaled_df.shape)
	print()
	print("Aligned Levels Matrix Shape:", aligned_levels.shape)
	print()
	
#	processed_path = r"D:\OneDrive\桌面\Mario Maker 2 Levels.csv"
#	aligned_df.to_csv(processed_path, index=False, encoding='utf-8-sig')
#	print("New dataset has been saved in:", processed_path)
	

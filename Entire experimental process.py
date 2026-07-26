import os
import copy
import time
import torch
import random
import pickle
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
try:
	matplotlib.use('TkAgg')
except Exception as e:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from sklearn.manifold import TSNE
from dataclasses import dataclass
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from Data_Preprocessing import get_processed_data
from torch.utils.data import WeightedRandomSampler

# Obtain the absolute path directory where the current script file is located
BASE_DIR = Path(__file__).resolve().parent

# =========================================
# ======== Parameter Configuration ========
# =========================================
@ dataclass
class config:
	device: str = "cuda" if torch.cuda.is_available() else "cpu"

	# **** Trigger branch drive switch ****
	use_ddpm: bool = True # True = DDPM will run, False = DCGAN will run

	# Basic hyperparameter
	latent_dim: int = 128
	batch_size: int = 64
	num_classes: int = 2
	num_tiles: int = 5
	grad_clip: float = 1.0
	lambda_gp: float = 5

	# DDPM + MODRL hyperparameter
	T: int = 80             # Denoising steps
	sample_steps: int = 80  # Sampling steps
	epochs: int = 80        # Training steps

	# DCGAN + MODRL hyperparameter
	n_critic: int = 2  # Discriminator steps
	lambda_gp: float = 5.0  # WGAN-GP steps

	def __post_init__(self):
		if not self.use_ddpm:
			self.epochs = 40 # Training steps

	# Path configuration
	data_path: str = str(BASE_DIR / "Mario Maker 2 Levels.csv")
	npz_path: str = str(BASE_DIR / "MM2_level_data.npz")

cfg = config()

# =================================================
# ==== Experience Replay Pool (for MODRL only) ====
# =================================================
class ReplayBuffer:
	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
	
	def push(self, state, action, reward, next_state):
		if len(self.buffer) >= self.capacity:
			self.buffer.pop(0)
		self.buffer.append((state, action, reward, next_state))
	
	def sample(self, batch_size):
		state, action, reward, next_state = zip(*random.sample(self.buffer, batch_size))
		return (torch.stack(state),
		        torch.tensor(action, dtype=torch.long),
		        torch.stack(reward),
		        torch.stack(next_state))
	
	def __len__(self):
		return len(self.buffer)

# ===============================
# ==== Define MM2 Level Data ====
# ===============================
class MarioLevelDataset(Dataset):
	def __init__(self, grids, conditions, df_types, labels):
		self.grids = torch.from_numpy(grids).float()
		self.conditions = torch.from_numpy(conditions).float()
		self.rewards = torch.from_numpy(df_types).float()
		self.labels = torch.tensor(labels, dtype=torch.long)
	
	def __getitem__(self, idx):
		return self.grids[idx], self.conditions[idx], self.rewards[idx], self.labels[idx]
	
	def __len__(self):
		return len(self.grids)

# ====================================================
# ==== Dependent Variable: Playability Indicators ====
# ====================================================
class PlayabilityIndicatorCNN(nn.Module):
	def __init__(self, num_classes_tiles=5, num_objectives=3):
		super(PlayabilityIndicatorCNN, self).__init__()
		self.conv = nn.Sequential(
			nn.Conv2d(num_classes_tiles, 32, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.MaxPool2d(2),
			nn.Conv2d(32, 64, kernel_size=3, padding=1),
			nn.ReLU(),
			nn.AdaptiveAvgPool2d((4, 4))
		)
		self.fc = nn.Sequential(
			nn.Linear(64 * 4 * 4, 128),
			nn.ReLU(),
			nn.Linear(128, num_objectives),
			nn.Sigmoid()
		)
	
	def forward(self, x):
		x = self.conv(x)
		x = torch.flatten(x, start_dim=1)
		return self.fc(x)

# ===========================================
# ==== Online Condition Optimizer: MODRL ====
# ===========================================
class MODRL(nn.Module):
	def __init__(self, state_dim, action_dim, num_objectives):
		super(MODRL, self).__init__()
		self.fc = nn.Sequential(
			nn.Linear(state_dim, 128),
			nn.ReLU(),
			nn.Linear(128, action_dim * num_objectives),
		)
		self.action_dim = action_dim
		self.num_obj = num_objectives
	
	def forward(self, x):
		out = self.fc(x)
		return out.view(-1, self.action_dim, self.num_obj)
	
	def select_action(self, state, weights, epsilon):
		if np.random.rand() < epsilon:
			return np.random.randint(self.action_dim)
		
		if state.dim() == 1:
			state = state.unsqueeze(0)
		
		with torch.no_grad():
			q_values_vector = self(state)
			q_values_scalar = torch.matmul(q_values_vector, weights)
			action = torch.argmax(q_values_scalar, dim=1).item()
			return action

# ==================================
# ==== Self-Attention Mechanism ====
# ==================================
class SelfAttention(nn.Module):
	def __init__(self, in_dim):
		super(SelfAttention, self).__init__()
		self.query = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
		self.key = nn.Conv2d(in_dim, in_dim // 8, kernel_size=1)
		self.value = nn.Conv2d(in_dim, in_dim, kernel_size=1)
		self.gamma = nn.Parameter(torch.zeros(1))  # Used to control the magnitude of the influence of attention
	
	def forward(self, x):
		batch, C, H, W = x.size()
		# Q, K, V projection
		q = self.query(x).view(batch, -1, H * W).permute(0, 2, 1)  # [B, HW, C/8]
		k = self.key(x).view(batch, -1, H * W)  # [B, C/8, HW]
		
		# Calculate the attention weight map
		attn = F.softmax(torch.bmm(q, k), dim=-1)  # [B, HW, HW]
		
		v = self.value(x).view(batch, -1, H * W)  # [B, C, HW]
		out = torch.bmm(v, attn.permute(0, 2, 1))  # [B, C, HW]
		
		out = out.view(batch, C, H, W)
		return self.gamma * out + x  # Residual connection ensures gradient stability

# ===================================
# ==== Invoking Module: RL Logic ====
# ===================================
class MODRL_MODULE:
	def _apply_action_to_cond(self, cond, action):
		new_cond = cond.clone()
		step_size = 0.05
		dim_to_change = action // 2
		direction = 1 if action % 2 == 0 else -1

		if cond.dim() == 1:
			new_cond[dim_to_change] += direction * step_size
		else:
			new_cond[:, dim_to_change] += direction * step_size

		return torch.clamp(new_cond, -1.0, 1.0)

	def update_agent(self, state, action, rewards, next_state, weights):
		self.agent.train()
		device = state.device
		action = action.to(device)
		q_values_vector = self.agent(state)
		batch_indices = torch.arange(q_values_vector.size(0), device=device)
		q_values_action = q_values_vector[batch_indices, action]
		current_q_scalar = torch.matmul(q_values_action, weights)

		with torch.no_grad():
			next_q_values_scalar = torch.matmul(self.target_agent(next_state), weights)
			max_next_q = torch.max(next_q_values_scalar, dim=1)[0]
			target_q = torch.matmul(rewards, weights) + 0.99 * max_next_q

		loss = F.mse_loss(current_q_scalar, target_q)
		self.agent_optimizer.zero_grad()
		loss.backward()
		self.agent_optimizer.step()

		for param, target_param in zip(self.agent.parameters(), self.target_agent.parameters()):
			target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
		return loss.item()

# ====================================================
# ==== Independent Variable 1: DCGAN Architecture ====
# ====================================================
def gumbel_softmax(logits, temperature=1.0):
	gumbels = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
	y = (logits + gumbels) / temperature
	return F.softmax(y, dim=1)

class MarioConditionalGenerator(nn.Module):
	def __init__(self, latent_dim, cond_dim, num_classes=2, num_tiles=5):
		super(MarioConditionalGenerator, self).__init__()
		self.label_emb = nn.Embedding(num_classes, 32)
		self.num_tiles = num_tiles
		
		# The total dimension integrating latent variables,
		# continuous conditions and discrete label embeddings (128 + data_dim + 32)
		input_dim = latent_dim + cond_dim + 32
		
		# Projection layer: Projects a one-dimensional vector onto a low-resolution 3D feature map
		self.init_channels = 128
		self.init_h = 3
		self.init_w = 25
		
		self.project = nn.Sequential(
			nn.Linear(input_dim, self.init_channels * self.init_h * self.init_w),
			nn.BatchNorm1d(self.init_channels * self.init_h * self.init_w),
			nn.ReLU(True)
		)
		
		# Deep deconvolution network: Gradually increase the spatial size while reducing the number of channels
		self.dcgan = nn.Sequential(
			# 1. (128, 3, 25) -> (64, 7, 50)
			# H: (3-1)*2 - 2*1 + 5 = 7
			# W: (25-1)*2 - 2*1 + 4 = 50
			nn.ConvTranspose2d(self.init_channels, 64, kernel_size=(5, 4), stride=2, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(True),
			
			SelfAttention(64),
			
			# 2. (64, 7, 50) -> (32, 14, 100)
			# 7*2=14, 50*2=100
			nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			
			nn.Conv2d(32, 32, kernel_size=(1, 7), padding=(0, 3)),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			
			# 3. (32, 14, 100) -> (5, 27, 200)
			# (14-1)*2 - 2*1 + 5 - 1 = 27
			# 100*2 - 2*1 + 4 - 1 = 200
			nn.ConvTranspose2d(32, 32, kernel_size=(3, 4), stride=2, padding=1),
			nn.BatchNorm2d(32),
			nn.ReLU(True),
			
			nn.Conv2d(32, num_tiles, kernel_size=(1, 7), padding=(0, 3)),
			nn.Upsample(size=(27, 200), mode='bilinear', align_corners=False),
		)
	
	def forward(self, z, cond, label, temperature=1.0):
		lbl_h = self.label_emb(label)
		z = z.view(z.size(0), -1)
		cond = cond.view(cond.size(0), -1)
		lbl_h = lbl_h.view(lbl_h.size(0), -1)
		
		# Concatenated feature vector
		idx_h = torch.cat((z, cond, lbl_h), dim=1)
		
		# DCGAN core forward propagation
		x = self.project(idx_h)
		x = x.view(-1, self.init_channels, self.init_h, self.init_w)
		logits = self.dcgan(x)  # Convolution upsampling
		
		return gumbel_softmax(logits)
	
class MarioConditionDiscriminator(nn.Module):
	def __init__(self, cond_dim, num_classes=2, num_tiles=5):
		super(MarioConditionDiscriminator,self).__init__()
		self.label_emb = nn.Embedding(num_classes, 32)
		self.num_tiles = num_tiles
		
		# An additional channel is used for line position encoding
		self.cond_net = nn.Sequential(
			nn.Linear(cond_dim, 64),
			nn.LeakyReLU(0.2),
		)

		self.conv = nn.Sequential(
			nn.Conv2d(num_tiles + 2, 32, kernel_size=3, padding=1),
			nn.LeakyReLU(0.2),
			nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
			nn.LeakyReLU(0.2),
		)
		
		self.pool = nn.AdaptiveAvgPool2d((14, 100))
		
		flatten_dim = 64 * 14 * 100
		input_dim = 64 + flatten_dim + 32
			
		self.fc = nn.Sequential(
			nn.Linear(input_dim, 128),
			nn.LeakyReLU(0.2),
			nn.Linear(128, 1)
		)
	
	def forward(self, x, cond, label):
		# x shape: (B, 5, 27, 200)
		B, C, H, W = x.shape
		
		# Row position channel: Normalize each row from 0 to 1
		row_idx = torch.linspace(0, 1, H, device=x.device)  # (27,)
		row_channel = row_idx.view(1, 1, H, 1).expand(B, 1, H, W)  # (B,1,27,200)
		
		# Column position channel: Normalize each column from 0 to 1
		col_idx = torch.linspace(0, 1, W, device=x.device)
		col_channel = col_idx.view(1, 1, 1, W).expand(B, 1, H, W)  # (B,1,27,200)
		
		x_with_pos = torch.cat([x, row_channel, col_channel], dim=1)  # (B,6,27,200)
		
		lbl_h = self.label_emb(label).view(label.size(0), -1)
		c_h = self.cond_net(cond).view(cond.size(0), -1)
		x_h = self.conv(x_with_pos)
		x_h = self.pool(x_h)
		x_h = x_h.reshape(x_h.size(0), -1)
		merged = torch.cat((c_h, x_h, lbl_h), dim=1)
		
		return self.fc(merged)
	
# DCGAN Gradient Penalty Mechanism
def compute_gradient_penalty(discriminator, real_samples, fake_samples, cond, label, device):
	alpha = torch.rand((real_samples.size(0), 1, 1, 1), device=device)
	interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
	d_interpolates = discriminator(interpolates, cond, label)
	fake = torch.ones((real_samples.size(0), 1), device=device, requires_grad=False)
	
	gradients = torch.autograd.grad(
		outputs=d_interpolates, inputs=interpolates, grad_outputs=fake,
		create_graph=True, retain_graph=True, only_inputs=True
	)[0]
	gradients = gradients.reshape(gradients.size(0), -1)
	gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
	return gradient_penalty

# ==========================================
# ==== Comparative Study 1: DCGAN+MODRL ====
# ==========================================
class DCGAN_MODRL_Framework(MODRL_MODULE):
	def __init__(self, modrl_agent, cgan_generator, latent_dim, device="cpu"):
		self.agent = modrl_agent.to(device)

		self.target_agent = copy.deepcopy(modrl_agent).to(device)
		self.target_agent.eval()
		self.tau = 0.005

		self.generator = cgan_generator.to(device)
		self.latent_dim = latent_dim
		self.device = device
		self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-4)
	
	def generate_step(self, current_cond, label, weights, epsilon=0.1):
		action = self.agent.select_action(current_cond, weights, epsilon)
		new_cond = self._apply_action_to_cond(current_cond, action)
		
		batch_size = new_cond.size(0) if new_cond.dim() > 1 else 1
		z = torch.randn(batch_size, self.latent_dim, device=self.device)
		cond_passed = new_cond.unsqueeze(0) if new_cond.dim() == 1 else new_cond
		cond_passed = cond_passed.detach()
		label_passed = label.unsqueeze(0) if label.dim() == 0 else label
		label_passed = label_passed.detach()
		
		was_training = self.generator.training
		self.generator.eval()
		
		with torch.no_grad():
			generated_logits = self.generator(z, cond_passed, label_passed)
			generated_level = torch.argmax(generated_logits, dim=1)
		
		if was_training:
			self.generator.train()
		return new_cond, action, generated_level, generated_logits

# ===================================================
# ==== Independent Variable 2: DDPM Architecture ====
# ===================================================
def forward_diffusion_process(x_0, t, alphas_cumprod_tensor):
	noise = torch.randn_like(x_0)
	sinal_amplitude = torch.sqrt(alphas_cumprod_tensor[t])[:, None, None, None]
	noise_intensity = torch.sqrt(1.0 - alphas_cumprod_tensor[t])[:, None, None, None]
	return sinal_amplitude * x_0 + noise_intensity * noise, noise

class Diffusion_Mario_CNN(nn.Module):
	def __init__(self, cond_dim, num_classes=2, num_tiles=5):
		super(Diffusion_Mario_CNN, self).__init__()
		self.label_emb = nn.Embedding(num_classes, 32)
		self.time_mlp = nn.Sequential(
			nn.Linear(1, 32),
			nn.ReLU(),
			nn.Linear(32, 32)
			)
		
		total_input_channels = 5 + 32 + cond_dim + 32
		
		# Phase 1: Complete resolution input
		self.enc1 = nn.Sequential(
			nn.Conv2d(total_input_channels, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # (27, 200) --> (13, 100)
		
		# Phase 2: Half-resolution input
		self.enc2 = nn.Sequential(
			nn.Conv2d(64, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # (13, 100) --> (6, 50)
		
		# Bottleneck layer
		self.bottleneck = nn.Sequential(
			nn.Conv2d(128, 256, kernel_size=3, padding=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, 256, kernel_size=3, padding=1),
			nn.ReLU()
		)
		
		# Decoder layer (upsample + residual fusion details)
		self.up1 = nn.Upsample(size=(13, 100), mode='bilinear', align_corners=False)
		self.dec1 = nn.Sequential(
			nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU(),
			nn.Conv2d(128, 128, kernel_size=3, padding=1),
			nn.BatchNorm2d(128),
			nn.ReLU()
		)
		
		# Phase 2: Restore to the original size (27, 200)
		self.up2 = nn.Upsample(size=(27, 200), mode='bilinear', align_corners=False)
		self.dec2 = nn.Sequential(
			nn.Conv2d(128 + 64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU(),
			nn.Conv2d(64, 64, kernel_size=3, padding=1),
			nn.BatchNorm2d(64),
			nn.ReLU()
		)
		
		# Output layer
		self.final_conv = nn.Conv2d(64, num_tiles, kernel_size=3, padding=1)
		
	def forward(self, x, t, cond, label):
		# Embedding time and conditions
		t_emb = self.time_mlp(t.float().unsqueeze(-1))
		t_spatial = t_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

		cond_spatial = cond[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

		lbl_emb = self.label_emb(label)
		lbl_spatial = lbl_emb[:, :, None, None].expand(-1, -1, x.shape[2], x.shape[3])

		x_input = torch.cat([x, t_spatial, cond_spatial, lbl_spatial], dim=1)
		
		# U-Net forward propagation network logic
		# 1. Encoder phase
		s1 = self.enc1(x_input)  # Feature Extraction (High resolution) -> Reserve as the first Skip Connection backup
		p1 = self.pool1(s1) # First time to reduce the size
		
		s2 = self.enc2(p1) # Feature Extraction (Medium resolution) -> Reserve as the first Skip Connection backup
		p2 = self.pool2(s2) # Second time to reduce the size
		
		# 2. Bottleneck phase
		b = self.bottleneck(p2) # The lowest-level global spatial receptive field
		
		# 3. Decoder phase
		d1 = self.up1(b) # Upsampling amplification
		d1 = torch.cat([d1, s2], dim=1) # The characteristics of resolution in splicing
		d1 = self.dec1(d1) # Feature fusion
		
		d2 = self.up2(d1) # Upsample and amplify back to the original size
		d2 = torch.cat([d2, s1], dim=1) # Stitch together the fine detail features of the original resolution
		d2 = self.dec2(d2) # Feature fusion
		
		return self.final_conv(d2) # (B, 5, 27, 200)

# =========================================
# ==== Comparative Study 2: DDPM+MODRL ====
# =========================================
class DDPM_MODRL_Framework(MODRL_MODULE):
	def __init__(self, modrl_agent, diffusion_model, alpha_cumprod, T, device="cpu"):
		self.agent = modrl_agent.to(device)

		self.target_agent = copy.deepcopy(modrl_agent).to(device)
		self.target_agent.eval()
		self.tau = 0.005

		self.diffusion_model = diffusion_model
		self.T = T
		self.device = device
		self.alphas_cumprod = alpha_cumprod.to(self.device)
		self._prepare_diffusion_constants(alpha_cumprod)
		self.agent_optimizer = torch.optim.Adam(self.agent.parameters(), lr=1e-4)
	
	def _prepare_diffusion_constants(self, alpha_cumprod):
		alpha_bar = alpha_cumprod
		alpha_bar_prev = torch.cat([torch.tensor([1.0], device=self.device), alpha_bar[:-1]])
		self.alphas = alpha_cumprod / alpha_bar_prev
		self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas).detach()

		betas = 1.0 - self.alphas
		self.posterior_variance = (betas * (1 - alpha_bar_prev) / (1 - alpha_bar)).clamp(min=1e-20)

	def _q_sample(self, x_0, t, noise):
		sqrt_alpha_bar = torch.sqrt(self.alphas_cumprod[t])[:, None, None, None]
		sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - self.alphas_cumprod[t])[:, None, None, None]
		return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

	def p_sample_loop(self, batch_size, cond, label):
		was_training = self.diffusion_model.training
		self.diffusion_model.eval()

		with (torch.no_grad()):
			x_t = torch.randn(batch_size, 5, 27, 200, device=self.device)

			sample_steps = min(cfg.sample_steps, self.T)

			step_indices = torch.linspace(self.T - 1, 0, sample_steps, dtype=torch.long, device=self.device)

			for t in step_indices:
				t = int(t.item())
				t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)

				predict_noise = self.diffusion_model(x_t, t_tensor, cond, label)

				beta_t = 1.0 - self.alphas[t]
				alpha_bar = self.alphas_cumprod[t]

				sqrt_alpha_bar = torch.sqrt(alpha_bar)
				sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

				# Recover x0
				pred_x0 = (x_t - sqrt_one_minus * predict_noise) / sqrt_alpha_bar
				pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

				# Posterior mean
				if t > 0:
					alpha_bar_prev = self.alphas_cumprod[t - 1]
					coef1 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar))
					coef2 = (torch.sqrt(self.alphas[t]) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
					mean = coef1 * pred_x0 + coef2 * x_t
				else:
					mean = pred_x0

				if t > 0:
					noise = torch.randn_like(x_t)
					var = torch.sqrt(self.posterior_variance[t].detach())
					x_t = mean + var * noise
				else:
					x_t = mean

			generated_probs = F.softmax(x_t, dim=1)

		if was_training:
			self.diffusion_model.train()
		return generated_probs

	def compute_diffusion_loss(self, x_0, cond, label):
		batch_size = x_0.size(0)
		t = torch.randint(0, self.T, (batch_size,), device=self.device).long()
		noise = torch.randn_like(x_0)
		x_t = self._q_sample(x_0, t, noise)
		pred_noise = self.diffusion_model(x_t, t, cond, label)

		# Noise Loss
		loss_noise = F.mse_loss(pred_noise, noise)

		return loss_noise

	def generate_step(self, current_cond, label, weights, epsilon=0.1):
		action = self.agent.select_action(current_cond, weights, epsilon)
		new_cond = self._apply_action_to_cond(current_cond, action)
		
		batch_size = new_cond.size(0) if new_cond.dim() > 1 else 1

		was_training = self.diffusion_model.training
		self.diffusion_model.eval()
		
		cond_passed = new_cond.unsqueeze(0) if new_cond.dim() == 1 else new_cond
		
		with torch.no_grad():
			x_t = torch.randn(batch_size, 5, 27, 200, device=self.device)

			sample_steps = min(cfg.sample_steps, self.T)
			step_indices = torch.linspace(self.T - 1, 0, sample_steps, dtype=torch.long, device=self.device)

			for t in step_indices:
				t = int(t.item())
				t_tensor = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
				predict_noise = self.diffusion_model(x_t, t_tensor, cond_passed, label)

				beta_t = 1.0 - self.alphas[t]

				alpha_bar = self.alphas_cumprod[t]

				sqrt_alpha_bar = torch.sqrt(alpha_bar)
				sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

				# Recover x0
				pred_x0 = (x_t - sqrt_one_minus * predict_noise) / sqrt_alpha_bar
				pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

				# Posterior mean
				if t > 0:
					alpha_bar_prev = self.alphas_cumprod[t - 1]
					coef1 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar))
					coef2 = (torch.sqrt(self.alphas[t]) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
					mean = coef1 * pred_x0 + coef2 * x_t
				else:
					mean = pred_x0

				if t > 0:
					noise_t = torch.randn_like(x_t)
					var_t = torch.sqrt(self.posterior_variance[t].detach())
					x_t = mean + var_t * noise_t
				else:
					x_t = mean

			generated_probs = F.softmax(x_t, dim=1)
			generated_level = torch.argmax(generated_probs, dim=1)
		
		if was_training:
			self.diffusion_model.train()
		return new_cond, action, generated_level, generated_probs
	
# ==========================================
# ==== Visualization of Level Thumbnail ====
# ==========================================
def visualize_level(final_grids, final_conds, final_labels, framework,
					generator, latent_dim, num_examples=4, device='cpu'):
	colors = ['#87CEEB', '#8B4513', '#808080', '#FFD700', '#FF0000']
	cmap = plt.matplotlib.colors.ListedColormap(colors)
	num_examples = min(num_examples, len(final_grids))
	if num_examples == 0: return
	
	random_indices = random.sample(range(len(final_grids)), num_examples)
	fig, axes = plt.subplots(num_examples, 2, figsize=(12, 2.5 * num_examples))
	fig.suptitle(f"Real vs Hybrid {'DDPM+MODRL' if config.use_ddpm else 'DCGAN+MODRL'} Generated Level",
				 fontsize=14, fontweight='bold')
	if num_examples == 1: axes = np.expand_dims(axes, axis=0)
	
	generator.eval()
	with torch.no_grad():
		for i, idx in enumerate(random_indices):
			axes[i, 0].imshow(final_grids[idx], cmap=cmap)
			axes[i, 0].set_title(f"Real Level Index {idx}", fontsize=9)
			
			cond = torch.tensor(final_conds[idx], dtype=torch.float32).unsqueeze(0).to(device)
			label = torch.tensor([final_labels[idx]], dtype=torch.long).to(device)
			
			if cfg.use_ddpm:
				x_t = torch.randn(1, 5, 27, 200, device=device)
				for t in reversed(range(cfg.T)):
					t_tensor = torch.full((1,), t, device=device, dtype=torch.long)
					predict_noise = generator(x_t, t_tensor, cond, label)

					beta_t = 1.0 - framework.alphas[t]
					alpha_bar = framework.alphas_cumprod[t]

					sqrt_alpha_bar = torch.sqrt(alpha_bar)
					sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

					# Recover x0
					pred_x0 = (x_t - sqrt_one_minus * predict_noise) / sqrt_alpha_bar
					pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

					# Posterior mean
					if t > 0:
						alpha_bar_prev = framework.alphas_cumprod[t - 1]
						coef1 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar))
						coef2 = (torch.sqrt(framework.alphas[t]) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
						mean = coef1 * pred_x0 + coef2 * x_t
					else:
						mean = pred_x0

					if t > 0:
						noise_t = torch.randn_like(x_t)
						var_t = torch.sqrt(framework.posterior_variance[t].detach())
						x_t = mean + var_t * noise_t
					else:
						x_t = mean

				generated_probs = F.softmax(x_t[0], dim=0)
				level_np = torch.argmax(generated_probs, dim=0).cpu().numpy()
			
			else:
				noise = torch.randn(1, latent_dim, device=device)
				cgan_output = generator(noise, cond, label, temperature=0.2)
				cgan_level = torch.argmax(cgan_output[0], dim=0)  # shape: (27, 200)
				level_np = cgan_level.cpu().numpy()

			axes[i, 1].imshow(level_np, cmap=cmap)
			axes[i, 1].set_title(f"Generated Level {idx}", fontsize=9)

			for j in range(2): axes[i, j].axis('off')
		plt.tight_layout()
		plt.show()

# ========================================================================
# ===== Compare the Playability Indicators of Real and Generated data ====
# ========================================================================
def compare_metrics_distribution(real_df, gen_df):
	real_df = real_df.copy()
	gen_df = gen_df.copy()
	real_df['Source'] = 'Real Data'
	gen_df['Source'] = 'Generated Data'
	
	# Merge data makes drawing more convenient
	metrics = ['satisfaction_rate', 'like_rate_norm', 'difficulty_score', 'playability_index']
	combined_df =pd.concat([real_df[metrics + ['Source']], gen_df[metrics + ['Source']]], axis=0)
	
	# Drawing
	fig, axes = plt.subplots(1, 4, figsize=(22, 5))
	for i, metric in enumerate(metrics):
		sns.violinplot(data=combined_df, x='Source', y=metric, hue='Source', ax=axes[i], palette="Pastel1",
					   inner="quartile", legend=False)
		axes[i].set_title(f'Distribution of {metric.replace("_", " ")}', fontsize=12, fontweight='bold')
		axes[i].set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
		axes[i].grid(True, linestyle=':', linewidth=0.6)
	
	plt.tight_layout()
	plt.show()

# =======================================
# ==== Compute local spatial entropy ====
# =======================================
def compute_spatial_entropy(level):
	values, counts = np.unique(level, return_counts=True)
	probs = counts / counts.sum()
	entropy = -np.sum(probs * np.log2(probs + 1e-12))
	return entropy

# ==============================
# ==== Pareto-optimal front ====
# ==============================
def pareto_front(points):
	is_pareto = np.ones(len(points), dtype=bool)

	for i in range(len(points)):
		for j in range(len(points)):
			if i == j:
				continue
			if np.all(points[j] >= points[i]) and np.any(points[j] > points[i]):
				is_pareto[i] = False
				break
	return is_pareto

# ===================================
# ==== Main program running zone ====
# ===================================
if __name__ == "__main__":
	print("Loading CSV & NPZ files...")
	df = pd.read_csv(cfg.data_path)
	
	# Calculate playability indicators
	df['satisfaction_rate'] = df['likes'] / (df['likes'] + df['boos'] + 1)
	like_rate = df['likes'] / (df['plays'] + 1)
	df['like_rate_norm'] = (like_rate - like_rate.min()) / (like_rate.max() - like_rate.min() + 1e-8)
	df['difficulty_score'] = np.exp(
		-((df['clears'] / (df['plays'] + 1) - 0.20) ** 2) / (2 * (0.15 ** 2)))
	df['playability_index'] = (
			(df['satisfaction_rate'] * 0.50) +
			(df['like_rate_norm'] * 0.30) +
			(df['difficulty_score'] * 0.20)
	)
	
	reward_cols = ['satisfaction_rate', 'like_rate_norm', 'difficulty_score']
	final_rewards = df[reward_cols].fillna(0).values
	
	npz_loader = np.load(cfg.npz_path, allow_pickle=True)
	levels, data_ids = npz_loader['levels'], npz_loader['data_ids']
	
	id_to_index_dict = {str(uid).strip(): idx for idx, uid in enumerate(data_ids)}
	grid_list, valid_df_indices = [], []
	
	for df_idx, rid in enumerate(df['data_id'].values):
		rid_str = str(rid).strip()
		if rid_str in id_to_index_dict:
			grid_list.append(levels[id_to_index_dict[rid_str]])
			valid_df_indices.append(df_idx)
		if len(grid_list) >= 5000:
			break
	
	if not grid_list:
		print("Warning: ID mismatch occurred. Switching to absolute sequence slicing.")
		found_count = min(len(levels), len(df), 5000)
		final_grids = levels[:found_count]
		df = df.iloc[:found_count].reset_index(drop=True)
	else:
		final_grids = np.stack(grid_list)
		df = df.iloc[valid_df_indices].reset_index(drop=True)
	
	if np.issubdtype(final_grids.dtype, np.str_) or np.issubdtype(final_grids.dtype, np.object_):
		final_grids = np.unique(final_grids, return_inverse=True)[1].reshape(final_grids.shape)
	final_grids = np.clip(final_grids.astype(np.int32), 0, 4)
	
	meta_cols = ['data_id', 'gamestyle', 'theme', 'type', 'is_subworld', 'has_beaten']
	cond_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in meta_cols + reward_cols]
	final_conds = MinMaxScaler(feature_range=(-1, 1)).fit_transform(df[cond_cols].fillna(0).values)
	
	# Right density: The non-empty tile ratio of the right half (columns 100-200) is normalized to [-1, 1]
	right_density = np.array([(g[:, 100:] > 0).mean() for g in final_grids])
	right_density_scaled = right_density * 2 - 1  # [0,1] → [-1,1]
	
	# Overall sparsity: The proportion of blank tiles, the high value after inversion = density
	sparsity = np.array([(g == 0).mean() for g in final_grids])
	density_scaled = (1 - sparsity) * 2 - 1  # [0,1] → [-1,1]
	
	# Sparse levels (blank > 70%), the weight is 3.0; High-density levels (blank < 15%), the weight is 2, the other is 1
	sample_weights = np.where(sparsity > 0.7, 3.0,
					 np.where(sparsity < 0.15, 2.0, 1.0))
	sample_weights = torch.tensor(sample_weights, dtype=torch.float)
	
	sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
	
	final_conds = np.hstack([final_conds, right_density_scaled.reshape(-1, 1), density_scaled.reshape(-1, 1)])
	data_dim = final_conds.shape[1]
	action_dim = data_dim * 2  # Dynamic binding of action space perfectly eliminates the risk of crossing boundaries
	
	print("\nStep 1: Pre-training Playability Indicator CNN (Evaluator)...")

	evaluator = PlayabilityIndicatorCNN(num_classes_tiles=5, num_objectives=3).to(config.device)
	eval_optimizer = optim.Adam(evaluator.parameters(), lr=1e-3)
	
	satisfaction_scores = final_rewards[:, 0]
	median_thresh = np.median(satisfaction_scores)
	adaptive_labels = (satisfaction_scores > median_thresh).astype(np.int64)
	
	eval_dataset = MarioLevelDataset(final_grids, final_conds, final_rewards, adaptive_labels)
	
	eval_loader = DataLoader(eval_dataset, batch_size=cfg.batch_size, sampler=sampler)
	
	evaluator.train()
	for _ in range(20):
		for grids_b, conds_b, rewards_b, labels_b in eval_loader:
			grids_b_oh = F.one_hot(grids_b.long(), num_classes=5).permute(0, 3, 1, 2).float().to(config.device)
			eval_optimizer.zero_grad()
			loss_ev = F.mse_loss(evaluator(grids_b_oh), rewards_b.to(config.device))
			loss_ev.backward()
			eval_optimizer.step()
	evaluator.eval()
	
	print("\nStep 2: Constructing Training Networks...")
	
	buffer = ReplayBuffer(capacity=3000)
	weights = torch.tensor([0.5, 0.3, 0.2], device=cfg.device)
	modrl_agent = MODRL(state_dim=data_dim, action_dim=action_dim, num_objectives=3).to(config.device)
	
	betas = torch.linspace(1e-4, 0.02, cfg.T).to(cfg.device)
	alphas = 1.0 - betas
	alphas_cumprod = torch.cumprod(alphas, dim=0)
	
	# A single-loop tracking dictionary
	history = {'d_loss': [], 'g_loss': [], 'rl_loss': [], 'epochs': [], 'weighted_reward': []}
	
	# DDPM Logic
	if cfg.use_ddpm:
		diffusion_model = Diffusion_Mario_CNN(cond_dim=data_dim).to(cfg.device)
		framework = DDPM_MODRL_Framework(modrl_agent, diffusion_model, alphas_cumprod, cfg.T, cfg.device)
		gen_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
		generator = diffusion_model
		discriminator = None
		disc_optimizer = None
	
	# DCGAN Logic
	else:
		generator = MarioConditionalGenerator(latent_dim=cfg.latent_dim, cond_dim=data_dim).to(cfg.device)
		discriminator = MarioConditionDiscriminator(cond_dim=data_dim).to(cfg.device)
		framework = DCGAN_MODRL_Framework(modrl_agent, generator, cfg.latent_dim, cfg.device)
		gen_optimizer = optim.Adam(generator.parameters(), lr=1e-4, betas=(0.0, 0.9))
		disc_optimizer = optim.Adam(discriminator.parameters(), lr=5e-5, betas=(0.0, 0.9))
		
	print(f"\nStep 3: Launching {'DDPM+MODRL' if cfg.use_ddpm else 'CDGAN+MODRL'} Unified Main Co-optimization Loop...")

	print("Warming up Replay Buffer...")
	while len(buffer) < 256:
		for real_grids, conds, _, labels in eval_loader:
			for b_idx in range(min(8, conds.size(0))):
				c_s = conds[b_idx].clone().to(cfg.device)
				lbl = labels[b_idx].unsqueeze(0).to(cfg.device)

				# Randomly select an action to fill in
				act = np.random.randint(action_dim)
				n_s = framework._apply_action_to_cond(c_s, act)
				buffer.push(c_s.cpu(), act, torch.zeros(3), n_s.cpu())

	for epoch in range(cfg.epochs):
		pbar = tqdm(enumerate(eval_loader), total=len(eval_loader), desc=f"Epoch {epoch + 1}/{cfg.epochs}")
		generator.train()
		if discriminator is not None:
			discriminator.train()
		
		epoch_d, epoch_g, epoch_rl, epoch_playability = [], [], [], []
		temperature = max(0.2, 1.0 - epoch * 0.04)

		rl_loss_val = 0.0

		for i, (real_grids, conds, rewards_b, labels) in pbar:
			batch_size = real_grids.size(0)
			
			real_grids_oh = F.one_hot(real_grids.long(), num_classes=5).permute(0, 3, 1, 2).float().to(cfg.device)
			real_grids_oh = real_grids_oh * 2.0 - 1.0

			conds, labels = conds.to(cfg.device), labels.to(cfg.device)
			discrete_labels = labels.long()
				
			# DDPM training path
			if cfg.use_ddpm:
				loss_diff = framework.compute_diffusion_loss(real_grids_oh, conds, discrete_labels)

				gen_optimizer.zero_grad()

				loss_diff.backward()

				torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)

				gen_optimizer.step()
				
				g_loss_val = loss_diff.item()
				epoch_g.append(g_loss_val)
			
			else:
			# DCGAN training path
				# Update the discriminator (cuts off the computational graph, significantly saving video memory)
				disc_optimizer.zero_grad()
				noise = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
				fake_grids = generator(noise, conds, discrete_labels, temperature).detach()
				
				real_validity = discriminator(real_grids_oh, conds, discrete_labels)
				fake_validity = discriminator(fake_grids, conds, discrete_labels)
				
				gp = compute_gradient_penalty(discriminator, real_grids_oh, fake_grids, conds,
											  discrete_labels, cfg.device)
				
				d_loss = -torch.mean(real_validity) + torch.mean(fake_validity) + cfg.lambda_gp * gp
				
				d_loss.backward()
				disc_optimizer.step()
				epoch_d.append(d_loss.item())
				
				# Update regularly the generator
				g_loss_val = 0.0
				if i % cfg.n_critic == 0:
					gen_optimizer.zero_grad()
					noise = torch.randn(batch_size, cfg.latent_dim, device=cfg.device)
					gen_grids = generator(noise, conds, discrete_labels, temperature)
					
					# GAN against loss
					g_loss = -torch.mean(discriminator(gen_grids, conds, discrete_labels))
					
					# gen_grids shape: (B, num_tiles, 27, 200), softmax probability
					col_diff = gen_grids[:, :, :, 1:] - gen_grids[:, :, :, :-1]  # (B, 5, 27, 199)
					continuity_loss = col_diff.pow(2).mean()
					
					g_loss_total = g_loss + 0.05 * continuity_loss
					g_loss_total.backward()
					torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=cfg.grad_clip)
					gen_optimizer.step()
					g_loss_val = g_loss_total.item()  # Record pure adversarial losses
					epoch_g.append(g_loss_val)
			
			# MODRL reinforcement learning environment interlocking and playback
			epsilon = max(0.1, 1.0 - (epoch / (cfg.epochs * 0.8)))

			# RL interaction collection is conducted only once every 10 batches, and the rest of the time is skipped
			if i % 5 ==0:
				for b_idx in range(min(8, batch_size)):
					current_state = conds[b_idx].clone()
					current_label = labels[b_idx].unsqueeze(0)

					new_state, action, gen_level, generated_probs  = framework.generate_step(
						current_state, current_label, weights, epsilon)

					with torch.no_grad():
						gen_level_discrete = torch.argmax(generated_probs, dim=1)
						gen_level_oh = F.one_hot(gen_level_discrete.long(),
												 num_classes=5).permute(0, 3, 1, 2).float().to(cfg.device)
						rewards_vector = torch.clamp(evaluator(gen_level_oh), 0.0, 1.0).squeeze(0)
						real_score = torch.matmul(rewards_vector, weights).item()
					buffer.push(current_state.cpu(), action, rewards_vector.detach().cpu(), new_state.cpu())
					epoch_playability.append(real_score)

				if len(buffer) >= 128:
					b_s, b_a, b_r, b_ns = buffer.sample(64)
					rl_loss_val = framework.update_agent(b_s, b_a, b_r, b_ns, weights)
					epoch_rl.append(rl_loss_val)
			
			if cfg.use_ddpm:
				pbar.set_postfix(Diff_Loss=f"{g_loss_val:.3f}", RL_Loss=f"{rl_loss_val:.3f}")
			
			else:
				pbar.set_postfix(D_Loss=f"{d_loss.item():.3f}", G_Loss=f"{g_loss_val:.3f}",
								 RL_Loss=f"{rl_loss_val:.3f}")
		
		# Accurately collect the average indicators of the current round
		history['d_loss'].append(np.mean(epoch_d) if epoch_d else 0)
		history['g_loss'].append(np.mean(epoch_g) if epoch_g else 0)
		history['rl_loss'].append(np.mean(epoch_rl) if epoch_rl else 0)
		history['epochs'].append(epoch + 1)
		history["weighted_reward"].append(float(np.mean(epoch_playability)) if epoch_playability else 0.5)

		torch.cuda.empty_cache()
	
	print(f"\n{'DDPM+MODRL' if cfg.use_ddpm else 'DCGAN+MODRL'} framework terminated perfectly!")

	# ==================================================
	# ==== Horizontal comparison of data collection ====
	# ==================================================
	framework_name = 'DDPM_MODRL' if cfg.use_ddpm else 'DCGAN_MODRL'
	print(f"Collecting core experimental data...")

	desktop_path = Path.home() / "Desktop"

	# Dynamically read the OneDrive environment variables of the Windows system
	onedrive_env = os.environ.get("OneDrive") or os.environ.get("OneDriveCommercial")

	if onedrive_env:
		# If the system has a OneDrive environment variable, try concatenating the Desktop path within it
		onedrive_desktop = Path(onedrive_env) / "Desktop"
		if onedrive_desktop.exists():
			desktop_path = onedrive_desktop
	else:
		# Manual check for Windows without configured environment variables
		possible_onedrive = Path.home() / "OneDrive" / "Desktop"
		if possible_onedrive.exists():
			desktop_path = possible_onedrive

	# Create an exclusive file path on desktop
	results_dir = desktop_path / "Experimental Results"
	results_dir.mkdir(parents=True, exist_ok=True)
	output_charts_dir = desktop_path / "Experimental Charts"
	output_charts_dir.mkdir(parents=True, exist_ok=True)

	# Merge the final file path
	file_path = results_dir / f"{framework_name}_results.pkl"

	collect_data = {
		"framework": framework_name,
		"history": history,
	}

	with open(file_path, "wb") as f:
		pickle.dump(collect_data, f)

	print(f"Experimental results has been saved to {file_path} successfully!")

	# ==============================================================
	# ==== Visualization of the double Y-axis convergence curve ====
	# ==============================================================
	fig, ax1 = plt.subplots(figsize=(10, 5))
	ax1.set_xlabel('Epochs', fontweight='bold')
	
	if cfg.use_ddpm:
		ax1.set_ylabel('Diffusion Loss (MSE)', color='black', fontweight='bold')
		line1 = ax1.plot(history['g_loss'], color='#ff7f0e', marker='o', linewidth=2, label='Diffusion MSE Loss')
		lines = line1
	
	else:
		ax1.set_ylabel('GAN Loss (D & G)', color='black', fontweight='bold')
		line1 = ax1.plot(history['d_loss'], color='#1f77b4', marker='o', linewidth=2, label='Discriminator Loss (D)')
		line2 = ax1.plot(history['g_loss'], color='#ff7f0e', linestyle='--', marker='s', linewidth=2,
						 label='Generator Loss (G)')
		lines = line1 + line2
	
	ax1.grid(True, linestyle=':', alpha=0.6)
	
	ax2 = ax1.twinx()
	ax2.set_ylabel('RL Loss (Q-Network)', color='#2ca02c', fontweight='bold')
	line3 = ax2.plot(history['rl_loss'], color='#2ca02c', linestyle='-.', marker='^', linewidth=2, label='MODRL Q Loss')
	ax2.tick_params(axis='y', labelcolor='#2ca02c')
	
	lines = lines + line3
	ax1.legend(lines, [l.get_label() for l in lines], loc='upper right')
	plt.title("Co-optimization Convergence: Loss Curves", fontsize=12, fontweight='bold', pad=15)
	plt.tight_layout()
	plt.show()
	
	# Output the thumbnail
	visualize_level(final_grids, final_conds, adaptive_labels, framework, generator, cfg.latent_dim, 4, cfg.device)
	
	print("\nStep 4: Generating samples to evaluate playability distribution...")
	
	# Prepare real_df
	real_df = df.copy()
	
	# Prepare gen_df
	generator.eval()
	evaluator.eval()

	with torch.no_grad():
		sample_size = min(128, len(final_conds))
		sample_indices = np.random.choice(len(final_conds), sample_size, replace=False)
		sample_conds = torch.tensor(final_conds[sample_indices], dtype=torch.float32).to(cfg.device)
		sample_labels = torch.tensor(adaptive_labels[sample_indices], dtype=torch.long).to(cfg.device)

		# DDPM Logic
		if cfg.use_ddpm:
			x_t = torch.randn(sample_size, 5, 27, 200, device=cfg.device)
			for t in reversed(range(cfg.T)):
				t_tensor = torch.full((sample_size,), t, device=cfg.device, dtype=torch.long)
				predict_noise = generator(x_t, t_tensor, sample_conds, sample_labels)

				beta_t = 1.0 - framework.alphas[t]
				alpha_bar = framework.alphas_cumprod[t]

				sqrt_alpha_bar = torch.sqrt(alpha_bar)
				sqrt_one_minus = torch.sqrt(1.0 - alpha_bar)

				# Recover x0
				pred_x0 = (x_t - sqrt_one_minus * predict_noise) / sqrt_alpha_bar
				pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)

				# Posterior mean
				if t > 0:
					alpha_bar_prev = framework.alphas_cumprod[t - 1]
					coef1 = (torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar))
					coef2 = (torch.sqrt(framework.alphas[t]) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar))
					mean = coef1 * pred_x0 + coef2 * x_t
				else:
					mean = pred_x0

				if t > 0:
					noise_t = torch.randn_like(x_t)
					var_t = torch.sqrt(framework.posterior_variance[t].detach())
					x_t = mean + var_t * noise_t
				else:
					x_t = mean

			gen_level_discrete = torch.argmax(x_t, dim=1)
			gen_oh_eval = F.one_hot(gen_level_discrete.long(), num_classes=5).permute(0, 3, 1, 2).float().to(cfg.device)

		# DCGAN Logic
		else:
			noise = torch.randn(sample_size, cfg.latent_dim, device=cfg.device)
			dcgan_logits = generator(noise, sample_conds, sample_labels, temperature=0.2)
			gen_level_discrete = torch.argmax(dcgan_logits, dim=1)
			gen_oh_eval = F.one_hot(gen_level_discrete.long(), num_classes=5).permute(0, 3, 1, 2).float().to(cfg.device)

		gen_rewards = torch.clamp(evaluator(gen_oh_eval), 0.0, 1.0).cpu().numpy()

		gen_rewards = np.clip(gen_rewards, 0, 1)

	gen_df = pd.DataFrame(gen_rewards, columns=['satisfaction_rate', 'like_rate_norm', 'difficulty_score'])

	gen_df['playability_index'] = (
			(gen_df['satisfaction_rate'] * 0.50) +
			(gen_df['like_rate_norm'] * 0.30) +
			(gen_df['difficulty_score'] * 0.20)
	)

	# Output the playability indicators
	compare_metrics_distribution(real_df, gen_df)
	print("Compare to real and generated playability indicators done.")

	# =====================================
	# ==== Spatial Entropy calculation ====
	# =====================================
	real_entropy = []
	generated_entropy = []

	# Randomly select 500 levels to calculate Spatial Entropy
	entropy_sample_size = min(500, len(final_grids))
	entropy_indices = np.random.choice(len(final_grids), entropy_sample_size, replace=False)

	eval_batch_size = 64
	generated_levels_list = []

	for idx_1 in range(0, entropy_sample_size, eval_batch_size):
		batch_indices = entropy_indices[idx_1:idx_1 + eval_batch_size]
		cond = torch.tensor(final_conds[batch_indices], dtype=torch.float32).to(cfg.device)
		label = torch.tensor(adaptive_labels[batch_indices], dtype=torch.long, device=cfg.device)

		with torch.no_grad():
			if cfg.use_ddpm:
				generated_probs = framework.p_sample_loop(len(batch_indices), cond, label)
				generated_level = torch.argmax(generated_probs, dim=1).cpu().numpy()
			else:
				z = torch.randn(len(batch_indices), cfg.latent_dim, device=cfg.device)
				generated_probs = generator(z, cond, label)
				generated_level = torch.argmax(generated_probs, dim=1).cpu().numpy()

		generated_levels_list.extend(generated_level)

	for i, idx in enumerate(entropy_indices):
		real_level = final_grids[idx]
		generated_level = generated_levels_list[i]

		real_entropy.append(compute_spatial_entropy(real_level))
		generated_entropy.append(compute_spatial_entropy(generated_level))

	# Visualization
	data_entropy = [real_entropy, generated_entropy]
	plt.figure(figsize=[6, 5])
	plt.boxplot(data_entropy, tick_labels=["Real", "Generated"], patch_artist=True)
	plt.ylabel("Spatial Entropy")
	plt.title("Spatial Entropy Distribution Comparison")
	plt.grid(axis='y', linestyle='--', alpha=0.7)
	plt.show()

	t_stat, p_value = scipy.stats.ttest_ind(real_entropy, generated_entropy, equal_var=False)

	print("\nSpatial Entropy Welch t-test")
	print(f"T statistic : {t_stat:.4f}")
	print(f"P-value     : {p_value:.6f}")

	# =======================================
	# ==== Visualization of pareto front ====
	# =======================================
	generated_rewards = gen_rewards
	mask = pareto_front(generated_rewards)
	pareto_ratio = np.mean(mask) * 100
	print(f"Pareto Ratio = {pareto_ratio:.2f}%")

	fig = plt.figure(figsize=[8, 6])
	ax = fig.add_subplot(111, projection='3d')
	ax.scatter(generated_rewards[:, 0], generated_rewards[:, 1], generated_rewards[:, 2], s=10, alpha=0.2,
			   label="All Levels")
	ax.scatter(generated_rewards[mask, 0], generated_rewards[mask, 1], generated_rewards[mask, 2], color="red",
			   s=30, label="Pareto Front")
	ax.set_xlabel("Satisfaction Rate")
	ax.set_ylabel("Like Rate Norm")
	ax.set_zlabel("Difficulty Score")
	plt.legend()
	plt.title("3D Pareto Front")
	plt.show()

	# =============================================================
	# ==== t-distributed Stochastic Neighbor Embedding (t-SNE) ====
	# =============================================================
	generated_levels = gen_level_discrete.cpu().numpy()

	real = final_grids[sample_indices]
	real = real.reshape(real.shape[0], -1)

	generated = generated_levels.reshape(generated_levels.shape[0], -1)

	all_data = np.vstack([real, generated])

	tsne = TSNE(n_components=2, perplexity=min(30, sample_size - 1), random_state=42)
	embeddings = tsne.fit_transform(all_data)

	plt.scatter(embeddings[:len(real), 0], embeddings[:len(real), 1], label='Real', alpha=0.5)
	plt.scatter(embeddings[len(real):, 0], embeddings[len(real):, 1], label='Generated', alpha=0.5)
	plt.legend()
	plt.title("Structural Diversity Visualization")
	plt.show()

	# =========================================================================
	# ==== Automated trigger detect whether the dual framework is complete ====
	# =========================================================================
	dcgan_path = results_dir / "DCGAN_MODRL_results.pkl"
	ddpm_path = results_dir / "DDPM_MODRL_results.pkl"

	if dcgan_path.exists() and ddpm_path.exists():
		print(f"Two experimental data of hybrid frameworks has been saved to {file_path}")
		output_charts_dir.mkdir(parents=True, exist_ok=True)

		# Read collected dual framework data
		with open(dcgan_path, "rb") as f:
			dcgan_data = pickle.load(f)
		with open(ddpm_path, "rb") as f:
			ddpm_data = pickle.load(f)

		# Configure chart style
		plt.rcParams.update({
			"font.family": "serif",
			"font.size": 11,
			"axes.labelsize": 12,
			"axes.titlesize": 13,
			"xtick.labelsize": 10,
			"ytick.labelsize": 10,
			"figure.titlesize": 14,
			"axes.grid": True,
			"grid.alpha": 0.3
		})

		# ===================================================
		# ==== Chart 1: Comparison of convergence curves ====
		# ===================================================
		fig, ax = plt.subplots(figsize=(7, 4.5), dpi=300)

		ax.plot(dcgan_data["history"]["epochs"], dcgan_data["history"]["weighted_reward"], label="DCGAN + MODRL",
				color="#1f77b4", linestyle="--", linewidth=2, marker="o",
				markevery=max(1, len(dcgan_data["history"]["epochs"]) // 10),)
		ax.plot(ddpm_data["history"]["epochs"], ddpm_data["history"]["weighted_reward"], label="DDPM + MODRL (Ours)",
			color="#ff7f0e", linestyle="-", linewidth=2,
			marker="s", markevery=max(1, len(ddpm_data["history"]["epochs"]) // 10),)

		ax.set_xlabel("Training Epochs")
		ax.set_ylabel("Average Playability Score")
		ax.set_title("Co-optimization Convergence Curve Comparison")
		ax.legend(loc="lower right")
		max_epochs = max(len(dcgan_data["history"]["epochs"]), len(ddpm_data["history"]["epochs"]))
		ax.set_xlim(1, max_epochs)
		ax.set_ylim(0, 1.0)

		chart_path = output_charts_dir / "framework_convergence_comparison.png"
		plt.tight_layout()
		plt.savefig(chart_path)
		plt.close()

		# =================================================
		# ==== Chart 2: Quantitative Performance Table ====
		# =================================================
		table_rows = []

		if dcgan_data is not None:
			score = np.mean(dcgan_data['history']['weighted_reward'][-5:])
			table_rows.append(["DCGAN + MODRL", f"{score:.2f}"])

		if ddpm_data is not None:
			score = np.mean(ddpm_data['history']['weighted_reward'][-5:])
			table_rows.append(["DDPM + MODRL", f"{score:.2f}"])

		if not table_rows:
			table_rows.append(["No Data Available", "0.0000"])

		fig, ax = plt.subplots(figsize=(6.5, 2.5), dpi=300)
		ax.axis('off')
		ax.axis('tight')

		table_columns = ["Hybrid Framework", "Average Weighted Reward"]
		visual_table = ax.table(cellText=table_rows, colLabels=table_columns, loc='center', cellLoc='center')
		visual_table.auto_set_font_size(False)
		visual_table.set_fontsize(10)
		visual_table.scale(1.0, 1.7)

		for (row, col), cell in visual_table.get_celld().items():
			if row == 0:
				cell.set_text_props(weight='bold')

		plt.title("Quantitative Performance Comparison Table", fontsize=11, fontweight='bold', pad=10)
		table_img_path = output_charts_dir / "framework_comparison_table.png"
		plt.savefig(table_img_path, bbox_inches='tight')
		plt.close()
		print(f"Visual Table Chart Image has been saved to: {table_img_path}")

		print(f"All cross-framework comparison charts are saved on your desktop!")
	else:
		print("\nWaiting for the other framework's experimental data to be generated...")
		print(f"Currently missing: {'DCGAN_MODRL_results.pkl' if cfg.use_ddpm else 'DDPM_MODRL_results.pkl'}")


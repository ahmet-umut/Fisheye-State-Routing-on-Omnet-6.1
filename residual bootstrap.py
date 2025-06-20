#!/usr/bin/env python3

import statistics
import math
import random

import numpy as np
from scipy.stats import t
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def main():
	# =========== 1) Read data (x, time) from extra_copy ===========
	data = []
	with open("A:\\Omnet\\tp\\reports\\extra_copy", "r") as f:
		for line in f:
			parts = line.strip().split()
			if len(parts) == 2:
				x_val = float(parts[0])
				t_val = float(parts[1])
				data.append((x_val, t_val))
	if not data:
		print("No valid data in extra_copy.")
		return
	"""
	data = [(inputs[i], outputs[i]) for i in range(len(inputs))]
	"""

	# Group data by distinct X
	grouped_data = {}
	for x_val, t_val in data:
		grouped_data.setdefault(x_val, []).append(t_val)

	distinct_x = sorted(grouped_data.keys())

	# =========== 2) Basic Stats: T-dist & Bootstrap for each distinct X ===========

	def t_dist_ci(values, alpha=0.05):
		"""Return (low, high) 95% CI for mean using t-dist."""
		n = len(values)
		if n < 2:
			return (None, None)
		mean_val = statistics.mean(values)
		stdev_val = statistics.stdev(values)
		df = n - 1
		t_crit = t.ppf(1 - alpha/2, df)
		se_mean = stdev_val / math.sqrt(n)
		margin = t_crit * se_mean
		return (mean_val - margin, mean_val + margin)

	def bootstrap_mean_ci(values, B=10000, alpha=0.05):
		"""Return (low, high) 95% CI for mean using basic percentile bootstrap."""
		n = len(values)
		if n < 2:
			return (None, None)
		means = []
		for _ in range(B):
			sample = random.choices(values, k=n)
			means.append(statistics.mean(sample))
		means.sort()
		lower_idx = int((alpha/2) * B)
		upper_idx = int((1 - alpha/2) * B)
		return (means[lower_idx], means[upper_idx])

	# Gather for plotting
	x_vals_for_plot = []
	mean_vals_for_plot = []
	t_low_for_plot = []
	t_high_for_plot = []
	b_low_for_plot = []
	b_high_for_plot = []

	for x_val in distinct_x:
		times = grouped_data[x_val]
		mean_val = statistics.mean(times)
		print(f"\nX = {x_val}, N = {len(times)}, Mean = {mean_val:.2f} ns")

		if len(times) >= 2:
			ci_t_low, ci_t_high = t_dist_ci(times)
			ci_b_low, ci_b_high = bootstrap_mean_ci(times, B=10000)

			print(f" T-dist 95% CI: [{ci_t_low:.2f}, {ci_t_high:.2f}]")
			print(f" Bootstrap 95% CI: [{ci_b_low:.2f}, {ci_b_high:.2f}]")

			x_vals_for_plot.append(x_val)
			mean_vals_for_plot.append(mean_val)
			t_low_for_plot.append(ci_t_low)
			t_high_for_plot.append(ci_t_high)
			b_low_for_plot.append(ci_b_low)
			b_high_for_plot.append(ci_b_high)
		else:
			print(" Not enough data for CI calculations.")

	# =========== 3) Neural Network Regression with Scaling ===========
	X_arr = np.array([d[0] for d in data]).reshape(-1,1)
	Y_arr = np.array([d[1] for d in data]).reshape(-1,1)

	X_scaler = StandardScaler()
	Y_scaler = StandardScaler()

	X_scaled = X_scaler.fit_transform(X_arr)
	Y_scaled = Y_scaler.fit_transform(Y_arr).ravel()

	nn_model = MLPRegressor(
		hidden_layer_sizes=(16, 8),
		activation='relu',
		solver='adam',
		max_iter=1000,
		#random_state=0
	)

	nn_model.fit(X_scaled, Y_scaled)
	Y_pred_scaled = nn_model.predict(X_scaled)
	Y_pred = Y_scaler.inverse_transform(Y_pred_scaled.reshape(-1,1)).ravel()

	residuals = Y_arr.ravel() - Y_pred

	# =========== 4) Residual Bootstrap Over Extended X Range ===========
	# Let's say we want to see what happens up to x=4 (or beyond).
	# We'll define x_fine as 1..4 in steps of 0.1, plus any distinct x's (like 2).
	x_min = min(distinct_x)
	x_max = max(distinct_x)
	extended_max = x_max*2
	extended_min = x_min/2
	x_fine = np.linspace(extended_min, extended_max, 20)

	x_all_for_boot = sorted(set(distinct_x).union(x_fine))

	predictions_by_x = {xv: [] for xv in x_all_for_boot}

	B = 200  # bootstrap iterations
	n_data = len(Y_arr)

	for _ in range(B):
		sampled_res = np.array(random.choices(residuals, k=n_data))
		Y_star = Y_pred + sampled_res  # original scale
		# Scale Y_star
		Y_star_scaled = Y_scaler.fit_transform(Y_star.reshape(-1,1)).ravel()

		# Keep X scaling from the original fit
		nn_boot = MLPRegressor(
			hidden_layer_sizes=(16,8),
			activation='relu',
			solver='adam',
			max_iter=1000,
			#random_state=0
		)
		nn_boot.fit(X_scaled, Y_star_scaled)

		for xv in x_all_for_boot:
			xv_scaled = X_scaler.transform(np.array([[xv]]))
			pred_scaled = nn_boot.predict(xv_scaled)
			pred_original = Y_scaler.inverse_transform(pred_scaled.reshape(-1,1)).ravel()[0]
			predictions_by_x[xv].append(pred_original)

	# Summarize predictions
	x_boot_list = []
	mean_boot_list = []
	lower_ci_list = []
	upper_ci_list = []

	for xv in x_all_for_boot:
		preds = predictions_by_x[xv]
		preds.sort()
		lower_idx = int(0.025*B)
		upper_idx = int(0.975*B)
		ci_low = preds[lower_idx]
		ci_high = preds[upper_idx]
		mean_pred = sum(preds)/len(preds)

		x_boot_list.append(xv)
		mean_boot_list.append(mean_pred)
		lower_ci_list.append(ci_low)
		upper_ci_list.append(ci_high)

	# =========== 5) Single-Plot Visualization ===========

	plt.figure(figsize=(8,6))
	#plt.title("Combined Confidence Intervals & NN Residual Bootstrap", fontsize=13, fontweight='bold')
	plt.title(title, fontsize=13, fontweight='bold')

	# A) T-dist & percentile bootstrap error bars for each distinct x
	#   We'll do two separate calls to errorbar with slight x offsets
	if x_vals_for_plot:
		x_arr = np.array(x_vals_for_plot)
		mean_arr = np.array(mean_vals_for_plot)

		# T-dist
		t_lower_errors = mean_arr - np.array(t_low_for_plot)
		t_upper_errors = np.array(t_high_for_plot) - mean_arr

		"""
		plt.errorbar(
			x_arr - 0.03, mean_arr,
			yerr=[t_lower_errors, t_upper_errors],
			fmt='o', color='blue', capsize=5, label="T-dist CI"
		)

		# Bootstrap
		b_lower_errors = mean_arr - np.array(b_low_for_plot)
		b_upper_errors = np.array(b_high_for_plot) - mean_arr

		plt.errorbar(
			x_arr + 0.03, mean_arr,
			yerr=[b_lower_errors, b_upper_errors],
			fmt='s', color='orange', capsize=5, label="Bootstrap CI"
		)
		"""

	# B) NN residual bootstrap: mean + 95% band
	x_boot_array = np.array(x_boot_list)
	mean_boot_array = np.array(mean_boot_list)
	lower_boot_array = np.array(lower_ci_list)
	upper_boot_array = np.array(upper_ci_list)

	# Sort them by X just in case
	sort_idx = np.argsort(x_boot_array)
	x_boot_array   = x_boot_array[sort_idx]
	mean_boot_array = mean_boot_array[sort_idx]
	lower_boot_array= lower_boot_array[sort_idx]
	upper_boot_array= upper_boot_array[sort_idx]

	# Plot the NN mean as a line
	plt.plot(
		x_boot_array, mean_boot_array, 'b-',
		label="NN Mean Prediction"
	)
	# Fill between lower & upper
	plt.fill_between(
		x_boot_array, lower_boot_array, upper_boot_array,
		color='blue', alpha=0.2, label="NN 95% confidence band"
	)

	dpl = "observed data"

	# C) Scatter original data
	X_data = [d[0] for d in data]
	Y_data = [d[1] for d in data]
	plt.scatter(X_data, Y_data, color='red', label=dpl)

	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend(loc="best")
	plt.grid(True)
	plt.show()

title = "Effect of Offered Load on Data Transmission Overhead \n (Confidence Intervals & Neural Network Regression)"
xlabel = "Offered Load (pps)"
ylabel = "Data Transmission Overhead (bits transmitted per delivered bit)"

inputs = [16, 8, 16, 16, 16, 18, 18, 34, 34, 6, 62, 62, 6, 6, 6, 6, 8, 10, 10, 64, 10, 10, 66]
outputs = [4968.51963767875, 6544.223841510872, 4871.00967528494, 4597.536338416148, 4022.728567087152, 
	     4993.714076555554, 3187.1300585017034, 1876.5219055970144, 1700.0317333979467, 
	     8261.337737114285, 1378.510310044837, 1226.0805037246203, 8351.950739285714, 
	     7038.119148878785, 8592.478412479173, 9481.481481000004, 7242.138726916665, 
	     5530.255238972221, 7900.4031413173025, 1169.4922621595392, 5220.626436217391, 
	     6982.963014962968, 1066.5135555554195]

if __name__ == "__main__":
	main()

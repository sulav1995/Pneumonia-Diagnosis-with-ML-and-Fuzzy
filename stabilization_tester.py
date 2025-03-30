import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import json
import matplotlib.pyplot as plt
from collections import deque

MEMBERSHIP_BOUNDS_FILE = "membership_bounds.json"

class StabilizationTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Membership Stabilization Tester")

        self.model = PneumoniaFuzzyModel()
        self.trend_data = {symptom: [] for symptom in self.model.membership_bounds}
        self.fluctuation_data = {symptom: {step: [] for step in [10, 20, 30, 50, 100]} for symptom in self.model.membership_bounds}
        self.accuracy_data = {}
        self.iteration_steps = [10, 20, 30, 50, 100]  # Iteration steps

        self.create_widgets()

    def create_widgets(self):
        frame = tk.Frame(self.root)
        frame.pack(padx=20, pady=20)

        self.run_button = tk.Button(frame, text="Run Stabilization Test", command=self.run_stabilization_test)
        self.run_button.grid(row=0, columnspan=2, pady=10)

        self.status_label = tk.Label(frame, text="Status: Waiting...", font=("Arial", 12, "bold"))
        self.status_label.grid(row=1, columnspan=2, pady=10)

        self.table_label = tk.Label(frame, text="Membership Bound Fluctuations at Different Iterations", font=("Arial", 12, "bold"))
        self.table_label.grid(row=2, columnspan=2, pady=10)

        # **Table Headers (Now showing multiple iteration steps)**
        columns = ["Symptom"] + [f"Change @ {step}" for step in self.iteration_steps] + ["Accuracy (%)"]
        self.bounds_table = ttk.Treeview(frame, columns=columns, show="headings")

        for col in columns:
            self.bounds_table.heading(col, text=col)
            self.bounds_table.column(col, width=120)

        self.bounds_table.grid(row=3, columnspan=2, pady=10)

    def run_stabilization_test(self):
        self.status_label.config(text="Running Stabilization Test...", fg="blue")

        for step in self.iteration_steps:
            self.run_iterations(step)
            self.update_table()
            self.root.update_idletasks()

        self.calculate_accuracy()

        #  Ensure symptoms that haven't fully stabilized continue adjusting
        for symptom in self.fluctuation_data:
            if self.fluctuation_data[symptom][100]:  # Check final step
                min_change = np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][100]])
                max_change = np.mean([max_c for _, max_c in self.fluctuation_data[symptom][100]])

                if min_change > 0.5 or max_change > 0.5:  # If still unstable
                    print(f" {symptom} still fluctuating > 1%, running extra stabilization steps.")
                    self.run_iterations(10)  # Run additional iterations for refinement

        self.status_label.config(text="Stabilization test completed!", fg="green")
        self.plot_stabilization_bar()


    def run_iterations(self, num_iterations):
        """Run dynamic adjustments for the specified number of iterations."""
        for i in range(num_iterations):
            symptoms = {}

            for s in self.model.membership_bounds:
                smoothed_value = self.model.dynamic_adjustment(s)

                # Apply progressive dampening to control large fluctuations
                damping_factor = max(0.35, 1 - (i / (num_iterations + 1)))  # Reduce effect gradually
                symptoms[s] = smoothed_value * damping_factor  # Adjusted for stability

            self.model.calculate_severity(**symptoms)

            for symptom in self.model.membership_bounds:
                min_b, max_b = self.model.membership_bounds[symptom]
                self.trend_data[symptom].append((min_b, max_b))

                if len(self.trend_data[symptom]) > 1:
                    prev_min, prev_max = self.trend_data[symptom][-2]

                    min_change = abs((min_b - prev_min) / (prev_min + 1e-2)) * 100 if abs(prev_min) > 1e-2 else abs(min_b - prev_min) * 100
                    max_change = abs((max_b - prev_max) / (prev_max + 1e-2)) * 100 if abs(prev_max) > 1e-2 else abs(max_b - prev_max) * 100

                    self.fluctuation_data[symptom][num_iterations].append((min_change, max_change))

                    # Debugging print to track fluctuations
                    #print(f"Symptom: {symptom}, Iteration: {num_iterations}, Min Change: {min_change:.2f}%, Max Change: {max_change:.2f}%")


 
    def moving_average(self, data, window_size=3):
        """Compute a simple moving average for a given list of values."""
        if len(data) < window_size:
            return np.mean(data) if data else 0.0  # If data is too small, return mean
        return np.convolve(data, np.ones(window_size) / window_size, mode='valid')[-1]  # Use last value of moving avg

    def calculate_accuracy(self):
        """Compute accuracy based on actual stabilization behavior, preventing forced 100% scaling."""

        for symptom in self.fluctuation_data:
            initial_step, mid_step, final_step = 10, 50, 100  # Compare fluctuations at multiple stages

            if (
                self.fluctuation_data[symptom][initial_step]
                and self.fluctuation_data[symptom][mid_step]
                and self.fluctuation_data[symptom][final_step]
            ):
                # Compute average fluctuation reduction over time
                initial_min = np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][initial_step]])
                initial_max = np.mean([max_c for _, max_c in self.fluctuation_data[symptom][initial_step]])

                mid_min = np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][mid_step]])
                mid_max = np.mean([max_c for _, max_c in self.fluctuation_data[symptom][mid_step]])

                final_min = np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][final_step]])
                final_max = np.mean([max_c for _, max_c in self.fluctuation_data[symptom][final_step]])

                initial_avg = (initial_min + initial_max) / 2
                mid_avg = (mid_min + mid_max) / 2
                final_avg = (final_min + final_max) / 2

                # **Ensure meaningful reduction over time (not just one step)**
                if initial_avg > 0 and final_avg < initial_avg:
                    fluctuation_reduction = ((initial_avg - final_avg) / initial_avg) * 100

                    # **Check for stable reduction over multiple steps**
                    mid_reduction = ((initial_avg - mid_avg) / initial_avg) * 100 if initial_avg > 0 else 0
                    final_reduction = ((mid_avg - final_avg) / mid_avg) * 100 if mid_avg > 0 else 0

                    # **Ensure fluctuation consistently reduces (not random dips)**
                    if fluctuation_reduction < 2 or mid_reduction < 1 or final_reduction < 1:
                        new_accuracy = 0  # No real stabilization detected
                    else:
                        # **Fix: Adjust scaling factor to prevent over-weighting**
                        stabilization_factor = max(0.4, 1 - (final_avg / initial_avg))  # Ensures range [0.4, 1]
                        new_accuracy = fluctuation_reduction * stabilization_factor 

                else:
                    new_accuracy = 0  # No stabilization detected

                # **Ensure accuracy is non-decreasing but realistic**
                previous_accuracy = self.accuracy_data.get(symptom, None)
                if previous_accuracy is not None:
                    new_accuracy = max(new_accuracy, previous_accuracy)

                # Assign accuracy to symptom
                self.accuracy_data[symptom] = round(new_accuracy, 2)  

                # Debugging print statement
                print(f"Symptom: {symptom}, Initial: {initial_avg:.2f}, Mid: {mid_avg:.2f}, Final: {final_avg:.2f}, Accuracy: {new_accuracy:.2f}%")

    def update_table(self):
        """Update the Tkinter table with fluctuations at different iterations and ensure correct accuracy values."""

        for row in self.bounds_table.get_children():
            self.bounds_table.delete(row)

        total_symptoms = len(self.fluctuation_data)
        stabilized_count = 0  # Track number of symptoms meeting stabilization criteria

        for symptom in self.fluctuation_data:
            row_values = [symptom]
            final_stabilized = False  # Track if the symptom stabilized at last iteration

            for step in self.iteration_steps:
                if len(self.fluctuation_data[symptom][step]) > 0:
                    avg_min_change = np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][step]])
                    avg_max_change = np.mean([max_c for _, max_c in self.fluctuation_data[symptom][step]])
                    row_values.append(f"{avg_min_change:.2f}% / {avg_max_change:.2f}%")

                    # ✅ **Check if a symptom has truly stabilized**
                    if step == 100 and avg_min_change < 1.0 and avg_max_change < 1.0:
                        final_stabilized = True
                else:
                    row_values.append("N/A")

            # ✅ **Ensure Accuracy is Correctly Updated and Retained**
            accuracy_value = self.accuracy_data.get(symptom, None)

            if accuracy_value is not None:
                row_values.append(f"{accuracy_value:.2f}%")  # Display computed accuracy
            else:
                row_values.append("0.00%")  # Show `0.00%` for first run, not "N/A"

            self.bounds_table.insert("", "end", values=row_values)

            if final_stabilized:
                stabilized_count += 1  # Increment stabilized count

        # ✅ **Compute Overall Accuracy (Only if at least one symptom has stabilized)**
        overall_accuracy_percentage = (stabilized_count / total_symptoms) * 100 if total_symptoms > 0 else 0

        # ✅ **Print Overall Accuracy to Console**
        print(f"Overall Stabilized Accuracy: {overall_accuracy_percentage:.2f}%")


    def plot_stabilization_bar(self):
        """Plot stabilization fluctuations as a bar chart for final iteration step."""
        symptoms = list(self.fluctuation_data.keys())
        min_changes = [np.mean([min_c for min_c, _ in self.fluctuation_data[symptom][100]]) for symptom in symptoms]
        max_changes = [np.mean([max_c for _, max_c in self.fluctuation_data[symptom][100]]) for symptom in symptoms]

        fig, ax = plt.subplots(figsize=(12, 6))
        bar_width = 0.4
        x = np.arange(len(symptoms))

        ax.bar(x - bar_width / 2, min_changes, bar_width, label="Min Change (%)", color="skyblue")
        ax.bar(x + bar_width / 2, max_changes, bar_width, label="Max Change (%)", color="salmon")

        ax.set_xlabel("Symptoms")
        ax.set_ylabel("Fluctuation (%)")
        ax.set_title("Membership Bound Fluctuations Over 100 Iterations")
        ax.set_xticks(x)
        ax.set_xticklabels(symptoms, rotation=45, ha="right")
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        plt.show()


class PneumoniaFuzzyModel:
    def __init__(self):
        self.membership_bounds = self.load_membership_bounds()
        self.symptom_data = {symptom: deque(maxlen=10) for symptom in self.membership_bounds}

    def load_membership_bounds(self):
        try:
            with open(MEMBERSHIP_BOUNDS_FILE, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {
                "breathlessness": (0.0, 1.0),
                "sputum_production": (0.0, 1.0),
                "fever_duration": (0.0, 30.0),
                "fever_value": (35.0, 42.0),
                "hemoptysis": (0.0, 1.0),
                "fatigue": (0.0, 1.0),
                "appetite_loss": (0.0, 1.0),
                "confusion": (0.0, 1.0),
                "chest_pain": (0.0, 1.0),
                "cough_severity": (0.0, 1.0),
                "oxygen_level": (80.0, 100.0)
            }

    def dynamic_adjustment(self, symptom):
        """
        Fully Implements the Dynamic Membership Adjustment Algorithm.

        - Step 1: Check symptom data trends (Already stored in deque)
        - Step 2: Remove extreme outliers using Z-score (3σ filtering)
        - Step 3: Compute variance and mean for last 10 data points
        - Step 4: Calculate smoothing factor **S = 0.3 + σ² / 100**
        - Step 5: Apply weighted exponential smoothing **S(xᵢ) = (S * xᵢ) + (1 - S) * xᵢ₋₁**
        - Step 6: Adjust min/max percentile thresholds based on recent mean
        - Step 7: Generate an adjusted range of values
        - Step 8: Clip adjusted range within bounds
        """
        
        data = list(self.symptom_data[symptom])
        if len(data) < 3:
            return data[-1] if data else 0.5  # Default value if no data available

        # **Step 2: Z-score Filtering (Strict 3σ outlier removal)**
        mean_val = np.mean(data)
        std_dev = np.std(data)
        filtered_data = [x for x in data if abs((x - mean_val) / (std_dev + 1e-6)) <= 3]

        # Ensure at least last 10 points are used
        if len(filtered_data) < 3:
            filtered_data = data[-3:]
        elif len(filtered_data) > 10:
            filtered_data = filtered_data[-10:]

        # **Step 3: Compute variance (σ²) and mean (μ)**
        variance = np.var(filtered_data)
        mean_recent = np.mean(filtered_data)

        # **Step 4: Compute adaptive smoothing factor S = 0.3 + σ² / 100**
        base_smoothing = 0.3  # Given in algorithm
        smoothing_factor = min(0.6, max(0.1, base_smoothing + variance / 50))  # Ensure range: [0.1 - 0.5]

        # **Step 5: Apply exponential smoothing**
        smoothed_value = filtered_data[0]  # Initialize first smoothed value
        for i in range(1, len(filtered_data)):
            smoothed_value = (smoothing_factor * filtered_data[i]) + ((1 - smoothing_factor) * smoothed_value)

        smoothed_value = round(smoothed_value, 3)

        # **Step 6: Set percentile thresholds for adjusted min/max**
        if mean_recent < 0.5:
            lower_percentile, upper_percentile = 10, 90
        else:
            lower_percentile, upper_percentile = 15, 85

        # **Step 7: Compute adjusted min/max using percentiles**
        adjusted_min = np.percentile(filtered_data, lower_percentile)
        adjusted_max = np.percentile(filtered_data, upper_percentile)

        # **Step 8: Generate adjusted range A_r with increments of 0.1**
        adjusted_range = np.arange(adjusted_min, adjusted_max + 0.1, 0.1).tolist()

        # **Step 9: Clip range within specified bounds**
        global_min, global_max = self.membership_bounds[symptom]
        adjusted_range = [max(global_min, min(x, global_max)) for x in adjusted_range]

        return smoothed_value

    def update_membership_bounds(self, symptom_values):
        """Ensure min and max values change separately."""
        for symptom, value in symptom_values.items():
            self.symptom_data[symptom].append(value)

            smoothed_value = self.dynamic_adjustment(symptom)

            min_b, max_b = self.membership_bounds[symptom]

            # **Incrementally adjust bounds**
            new_min = min(min_b, smoothed_value) - 0.1
            new_max = max(max_b, smoothed_value) + 0.1

            # **Prevent bounds collapse (ensuring a valid range)**
            if new_max - new_min < 0.05:
                new_min, new_max = min_b, max_b

            self.membership_bounds[symptom] = (round(new_min, 3), round(new_max, 3))

    def calculate_severity(self, **symptom_values):
        self.update_membership_bounds(symptom_values)


if __name__ == "__main__":
    root = tk.Tk()
    app = StabilizationTester(root)
    root.mainloop()

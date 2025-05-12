import json
import matplotlib.pyplot as plt
import os

# Filepath to the JSON file
json_file_path = "/home/caio.dasilva/Vibe3D/kitwarebrc/marks2302.json"

# Read the JSON file
with open(json_file_path, "r") as file:
    data = json.load(file)

# Count occurrences of each category
counts = {"correct": 0, "incorrect": 0, "two_subjects": 0}
for key, value in data.items():
    if value in counts:
        counts[value] += 1

# Ensure the 'charts' directory exists
charts_dir = "charts"
os.makedirs(charts_dir, exist_ok=True)

# Generate a pie chart
labels = counts.keys()
sizes = counts.values()
colors = ["#4CAF50", "#F44336", "#FFC107"]  # Green, Red, Yellow
explode = (0.1, 0, 0)  # Highlight the "correct" slice

def autopct_with_total(pct, all_vals):
    total = sum(all_vals)
    absolute = int(round(pct * total / 100.0))
    return f"{pct:.1f}%\n({absolute})"

plt.figure(figsize=(8, 6))
plt.pie(
    sizes, 
    explode=explode, 
    labels=labels, 
    colors=colors, 
    autopct=lambda pct: autopct_with_total(pct, sizes), 
    startangle=140
)
plt.title("BRC2 - Ground Truth Images")
plt.savefig(os.path.join(charts_dir, "category_distribution_pie_chart.png"))
plt.close()

# Generate a bar chart
plt.figure(figsize=(8, 6))
plt.bar(labels, sizes, color=colors)
plt.xlabel("Category")
plt.ylabel("Count")
plt.title("Count of Each Category")
plt.savefig(os.path.join(charts_dir, "category_distribution_bar_chart.png"))
plt.close()

print("Charts saved in the 'charts' folder as 'category_distribution_pie_chart.png' and 'category_distribution_bar_chart.png'.")
import matplotlib.pyplot as plt
import math

# Define the data
usage_kernels = [164, 165, 168, 177, 192, 553, 581, 599, 605]
y_values = [1, 1, 1, 1, 1, 1, 1, 1, 1]  # Arbitrary y-values for illustration

# Define offload opportunities
offload_opportunities = [
    # (Idle Cycles, Offload Type, Arrival TF, Transfer Back TF)
    (0.00, None, None, None),
    (82613451.60, None, None, None),
    (1032121971.60, 'SSD', 169, 176),
    (1956419172.00, 'SSD', 178, 191),
    (33098235207.60, 'SSD', 193, 552),
    (461603617.20, 'CPU', 554, 580),
    (279893602.80, 'CPU', 582, 598),
    (95867288.40, None, None, None)
]

# Define regions of interest
regions = [
    {
        'start': 160,
        'end': 200,
        'data_x': [xi for xi in usage_kernels if 160 <= xi <= 200],
        'data_y': [yi for xi, yi in zip(usage_kernels, y_values) if 160 <= xi <= 200]
    },
    {
        'start': 540,
        'end': 610,
        'data_x': [xi for xi in usage_kernels if 540 <= xi <= 610],
        'data_y': [yi for xi, yi in zip(usage_kernels, y_values) if 540 <= xi <= 610]
    }
]

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True, figsize=(12, 3))

# Plot data for the first region (160-200)
ax1.plot(regions[0]['data_x'], regions[0]['data_y'], 'bo', label='Used Kernels')
ax1.set_xlim(regions[0]['start'], regions[0]['end'])
ax1.set_xlabel('Kernel (Timeframe ID)')
ax1.set_title('160 to 200')
ax1.get_yaxis().set_visible(False)

# Plot data for the second region (550-610)
ax2.plot(regions[1]['data_x'], regions[1]['data_y'], 'bo', label='Used Kernels')
ax2.set_xlim(regions[1]['start'], regions[1]['end'])
ax2.set_xlabel('Kernel (Timeframe ID)')
ax2.set_title('550 to 610')
ax2.get_yaxis().set_visible(False)

# Hide the y-axis labels on the second subplot
plt.setp(ax2.get_yticklabels(), visible=False)

# Add diagonal lines to indicate the break
d = .015  # How big to make the diagonal lines in axes coordinates
kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
# Top-right diagonal line
ax1.plot((1 - d, 1 + d), (-d, +d), **kwargs)
# Bottom-right diagonal line
ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)

kwargs.update(transform=ax2.transAxes)  # Switch to the second axes
# Top-left diagonal line
ax2.plot((-d, +d), (-d, +d), **kwargs)
# Bottom-left diagonal line
ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)

# Add a main title
fig.suptitle('Tensor 995 Usage and Offload Opportunities', fontsize=16)

# Adjust layout to accommodate the main title
plt.tight_layout(rect=[0, 0.03, 1, 0.95])

# Function to add vertical lines for offload events
def add_offload_lines(ax, arrival, transfer_back, offload_type):
    if offload_type == 'CPU':
        color = 'green'
    elif offload_type == 'SSD':
        color = 'purple'
    else:
        return  # Do nothing for None
    
    # Plot arrival line (dotted)
    ax.axvline(x=arrival, color=color, linestyle=':', linewidth=2)
    
    # Plot transfer back line (dotted)
    ax.axvline(x=transfer_back, color=color, linestyle=':', linewidth=2)

# Iterate through offload opportunities and add lines
for idx, (idle_cycles, offload_type, arrival_tf, transfer_back_tf) in enumerate(offload_opportunities):
    if offload_type is None:
        continue  # Skip if there's no offload
    # Determine which subplot the arrival_tf belongs to
    if regions[0]['start'] <= arrival_tf <= regions[0]['end']:
        add_offload_lines(ax1, arrival_tf, transfer_back_tf, offload_type)
    elif regions[1]['start'] <= arrival_tf <= regions[1]['end']:
        add_offload_lines(ax2, arrival_tf, transfer_back_tf, offload_type)

# Create a custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Used Kernels', markerfacecolor='b', markersize=8),
    Line2D([0], [0], color='green', linestyle=':', lw=2, label='CPU Offload'),
    Line2D([0], [0], color='purple', linestyle=':', lw=2, label='SSD Offload'),
    Line2D([0], [0], linestyle='--', color='k', lw=1, label='Break')
]
fig.legend(handles=legend_elements, loc='upper right')

output_file = f'tensor_995_usage_offload.png'
plt.savefig(output_file)
plt.show()
print(f"Saved tensor usage and offload opportunities plot to {output_file}")
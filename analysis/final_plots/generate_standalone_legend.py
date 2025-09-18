# analysis/final_plots/generate_standalone_legend.py

import os
import matplotlib.pyplot as plt
import argparse
from matplotlib.lines import Line2D

# --- Final Plot Style Guide ---
# This is our single source of truth for visual styles, copied from the other
# final plotting scripts to ensure perfect consistency.
FINAL_PLOT_STYLES = {
    "mmar":            {"label": "MMAR",       "color": "#e41a1c", "marker": "X"},
    "sakura-animal":   {"label": "SAKURA Animal",   "color": "#377eb8", "marker": "o"},
    "sakura-emotion":  {"label": "SAKURA Emotion",  "color": "#4daf4a", "marker": "v"},
    "sakura-gender":   {"label": "SAKURA Gender",   "color": "#ff7f00", "marker": "s"},
    "sakura-language": {"label": "SAKURA Language", "color": "#984ea3", "marker": ">"}
}

def generate_legend(output_dir: str, base_filename: str):
    """
    Generates and saves a standalone legend image.
    """
    print(f"\n--- Generating Standalone Legend ---")
    
    # --- 1. Create Custom Legend Handles ---
    # We manually create a list of 'handles' (the visual elements) and 'labels' (the text)
    # to have full control over the legend's appearance.
    handles = []
    labels = []
    for dataset_name, style in FINAL_PLOT_STYLES.items():
        labels.append(style['label'])
        handle = Line2D([0], [0], # Placeholder data points
                        color=style['color'],
                        marker=style['marker'],
                        linestyle='-',
                        linewidth=3,  # Slightly thicker line for clarity
                        markersize=12) # Slightly larger marker for clarity
        handles.append(handle)

    # --- 2. Create a Figure Specifically for the Legend ---
    # The figsize is chosen to be wide and short, suitable for a horizontal legend.
    legend_fig = plt.figure(figsize=(15, 2))
    
    # Create the legend on the figure canvas itself, not inside any axes.
    legend = legend_fig.legend(
        handles=handles, 
        labels=labels,
        loc='center',      # Place it in the center of the figure
        ncol=len(labels),  # Arrange all items in a single row
        frameon=False,     # Remove the box around the legend
        fontsize=24        # Use a large, readable font
    )

    # --- 3. Hide the Empty Plot Axes ---
    # A new figure has an empty plot by default. We must turn it off.
    plt.gca().axis('off')

    # --- 4. Save the Output Files ---
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the PNG version
    png_path = os.path.join(output_dir, f"{base_filename}.png")
    # 'bbox_inches="tight"' is the key command that crops the saved image
    # to the bounding box of the legend itself.
    legend_fig.savefig(png_path, bbox_inches='tight', dpi=300)
    print(f"  - Legend PNG saved to: {png_path}")

    # Save the PDF version
    pdf_path = os.path.join(output_dir, f"{base_filename}.pdf")
    legend_fig.savefig(pdf_path, bbox_inches='tight', format='pdf')
    print(f"  - Legend PDF saved to: {pdf_path}")
    
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a standalone legend for the final plots.")
    parser.add_argument('--plots_dir', type=str, default='./final_plots', help="The directory where the legend will be saved.")
    parser.add_argument('--filename', type=str, default='standalone_legend', help="The base filename for the output legend files.")
    args = parser.parse_args()
    
    generate_legend(args.plots_dir, args.filename)
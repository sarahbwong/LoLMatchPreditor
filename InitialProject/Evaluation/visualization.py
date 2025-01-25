import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibrationDisplay

def plot_roc(fpr, tpr):
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC Curve for Overall Win Probability Model')
    plt.grid(True)

def plot_calibration_from_predictions(y_true, predicted_probs, names, title="Calibration Plots"):
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.get_cmap("Dark2")
    
    # Create main calibration curve plot
    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    
    for i, (y_prob, name) in enumerate(zip(predicted_probs, names)):
        display = CalibrationDisplay.from_predictions(
            y_true,
            y_prob,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title(title)
    
    # Add histogram plots
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, name in enumerate(names):
        if i >= len(grid_positions):  # Ensure no out-of-bounds
            break
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")
    
    plt.tight_layout()
    plt.show()
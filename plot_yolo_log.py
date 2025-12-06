from pathlib import Path
import re
import matplotlib.pyplot as plt

# Path to your saved log file
LOG_PATH = Path("training_log.txt")

# Regex for the epoch / loss line, e.g.:
# "      1/100      2.01G     0.9992      11.41      1.385         27        640: ..."
epoch_line_re = re.compile(
    r"^\s*(\d+)\s*/\s*\d+\s+"      # epoch number like "1/100"
    r"([\d.]+)G\s+"                # GPU_mem (ignored except for check)
    r"([\d.]+)\s+"                 # box_loss
    r"([\d.]+)\s+"                 # cls_loss
    r"([\d.]+)"                    # dfl_loss
)

# Regex for the metrics "all" line, e.g.:
# "                   all        191        326      0.772      0.265      0.348      0.213"
metrics_line_re = re.compile(
    r"^\s*all\s+\d+\s+\d+\s+"
    r"([\d.]+)\s+"                 # P
    r"([\d.]+)\s+"                 # R
    r"([\d.]+)\s+"                 # mAP50
    r"([\d.]+)"                    # mAP50-95
)

epochs = []
box_losses = []
cls_losses = []
dfl_losses = []

# Optional: metrics
precisions = []
recalls = []
map50s = []
map5095s = []

with LOG_PATH.open("r", encoding="utf-8") as f:
    for line in f:
        # Try to match epoch / loss line
        m_epoch = epoch_line_re.match(line)
        if m_epoch:
            epoch = int(m_epoch.group(1))
            box_loss = float(m_epoch.group(3))
            cls_loss = float(m_epoch.group(4))
            dfl_loss = float(m_epoch.group(5))

            epochs.append(epoch)
            box_losses.append(box_loss)
            cls_losses.append(cls_loss)
            dfl_losses.append(dfl_loss)
            continue

        # Try to match metrics line
        m_metrics = metrics_line_re.match(line)
        if m_metrics:
            P = float(m_metrics.group(1))
            R = float(m_metrics.group(2))
            mAP50 = float(m_metrics.group(3))
            mAP5095 = float(m_metrics.group(4))

            precisions.append(P)
            recalls.append(R)
            map50s.append(mAP50)
            map5095s.append(mAP5095)
            continue

# ---- Plot losses ----
plt.figure()
plt.plot(epochs, box_losses, label="box_loss")
plt.plot(epochs, cls_losses, label="cls_loss")
plt.plot(epochs, dfl_losses, label="dfl_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Losses per Epoch")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ---- Optional: plot metrics if we got them ----
if map50s:
    # Note: metrics are usually once per epoch; if lengths mismatch,
    # you might want to align them manually.
    plt.figure()
    plt.plot(epochs[:len(map50s)], map50s, label="mAP50")
    plt.plot(epochs[:len(map5095s)], map5095s, label="mAP50-95")
    plt.xlabel("Epoch")
    plt.ylabel("Metric value")
    plt.title("Validation Metrics per Epoch")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

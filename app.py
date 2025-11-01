import gradio as gr
import torch
from model import run_inference

# --- model artifact paths (inside repo) ---
CKPT_PATH = "gegnn_best.pt"
STATS_PATH = "norm_stats.pt"

def classify(csv_file):
    try:
        result = run_inference(csv_file.name, CKPT_PATH, STATS_PATH)
        frauds = result["fraudulent_reviews"]

        if len(frauds) == 0:
            return "No fraudulent reviews detected."

        output = "\n\n".join([
            f"Reviewer: {r['reviewerID']}\nReview: {r['reviewText']}"
            for r in frauds
        ])
        return output

    except Exception as e:
        return f"Error during inference: {str(e)}"

iface = gr.Interface(
    fn=classify,
    inputs=gr.File(label="Upload CSV file of reviews"),
    outputs="text",
    title="GE-GNN Fraud Detection",
    description=(
        "Upload a review dataset subset (Amazon or YelpChi format). "
        "The GE-GNN model runs graph-based fraud detection and lists "
        "reviews predicted as fraudulent."
    )
)

if __name__ == "__main__":
    iface.launch()

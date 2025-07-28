import gradio as gr
import json
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from train_utils import preprocess_input_for_prediction

def load_predictor_config(config_path="trained_predictors.json", benchmark_path="benchmark_data_Qwen_Qwen3-4B_TP_1_PP_1.json"):
    """Load the trained predictor configuration and benchmark metadata from JSON files."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Try to load benchmark metadata
        try:
            with open(benchmark_path, 'r') as f:
                benchmark_data = json.load(f)
                config["metadata"] = benchmark_data.get("metadata", {})
        except (FileNotFoundError, json.JSONDecodeError):
            # If benchmark file is not found, continue without metadata
            config["metadata"] = {}
            
        return config
    except FileNotFoundError:
        return {"error": "Config file not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

def predict_latency(batch_size, input_len, config, stage="prefill"):
    """Predict latency using the linear regression model."""
    if "error" in config:
        return f"Error: {config['error']}"
    
    try:
        # Get model parameters
        model_config = config["test_config"][stage]
        weights = np.array(model_config["weights"])
        bias = model_config["bias"]
        
        # Preprocess input features
        features = preprocess_input_for_prediction(batch_size, input_len, "gpu", stage)
        features_array = np.array(features)
        
        # Make prediction
        prediction = np.dot(weights, features_array) + bias
        prediction_ms = prediction * 1000
        
        return f"{prediction:.6f} seconds ({prediction_ms:.3f} ms)"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def create_latency_plot(config, max_batch_size=64, max_input_len=32768, unit_mode="seconds"):
    """Create interactive visualization plots for prefill and decode latency using Plotly."""
    # Determine unit conversion and labels
    unit_multiplier = 1000 if unit_mode == "milliseconds" else 1
    unit_label = "ms" if unit_mode == "milliseconds" else "seconds"
    unit_format = ".3f" if unit_mode == "milliseconds" else ".6f"
    if "error" in config:
        # Create error plot
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Prefill Latency", "Decode Latency"),
            horizontal_spacing=0.15
        )
        
        # Add error text annotations
        fig.add_annotation(
            text=f"Error: {config['error']}",
            xref="x", yref="y",
            x=0.5, y=0.5,
            showarrow=False,
            row=1, col=1
        )
        fig.add_annotation(
            text=f"Error: {config['error']}",
            xref="x2", yref="y2",
            x=0.5, y=0.5,
            showarrow=False,
            row=1, col=2
        )
        
        fig.update_layout(height=500, showlegend=False)
        return fig
    
    # Create batch size range
    batch_sizes = [bs for bs in [1, 2, 4, 8, 16, 32, 64] if bs <= max_batch_size]
    input_lens = [il for il in [256, 512, 1024, 2048, 4096, 8192, 16384, 32768] if il <= max_input_len]
    
    # Create subplots
    fig = sp.make_subplots(
        rows=1, cols=2,
        subplot_titles=(f"Prefill Latency vs Batch Size ({unit_label})", f"Decode Latency vs Batch Size ({unit_label})"),
        horizontal_spacing=0.15
    )
    
    # Color palette for different input lengths
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
    
    # Plot prefill latency
    for i, input_len in enumerate(input_lens):
        prefill_latencies = []
        hover_texts = []
        
        for bs in batch_sizes:
            features = preprocess_input_for_prediction(bs, input_len, "gpu", "prefill")
            weights = np.array(config["test_config"]["prefill"]["weights"])
            bias = config["test_config"]["prefill"]["bias"]
            latency = np.dot(weights, features) + bias
            latency_display = latency * unit_multiplier
            prefill_latencies.append(latency_display)
            
            # Create detailed hover text
            latency_ms = latency * 1000
            hover_texts.append(
                f"Batch Size: {bs}<br>"
                f"Input Length: {input_len}<br>"
                f"Prefill Latency: {latency:.6f}s ({latency_ms:.3f}ms)<br>"
                f"Features: {features}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=prefill_latencies,
                mode='lines+markers',
                name=f'Input Len: {input_len}',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=8, symbol='circle'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                legendgroup='prefill'
            ),
            row=1, col=1
        )
    
    # Plot decode latency
    for i, input_len in enumerate(input_lens):
        decode_latencies = []
        hover_texts = []
        
        for bs in batch_sizes:
            features = preprocess_input_for_prediction(bs, input_len, "gpu", "decode")
            weights = np.array(config["test_config"]["decode"]["weights"])
            bias = config["test_config"]["decode"]["bias"]
            latency = np.dot(weights, features) + bias
            latency_display = latency * unit_multiplier
            decode_latencies.append(latency_display)
            
            # Create detailed hover text
            latency_ms = latency * 1000
            hover_texts.append(
                f"Batch Size: {bs}<br>"
                f"Input Length: {input_len}<br>"
                f"Decode Latency: {latency:.6f}s ({latency_ms:.3f}ms)<br>"
                f"Features: {features}"
            )
        
        fig.add_trace(
            go.Scatter(
                x=batch_sizes,
                y=decode_latencies,
                mode='lines+markers',
                name=f'Input Len: {input_len}',
                line=dict(color=colors[i % len(colors)]),
                marker=dict(size=8, symbol='square'),
                hovertemplate='%{hovertext}<extra></extra>',
                hovertext=hover_texts,
                legendgroup='decode',
                showlegend=False  # Hide legend for decode to avoid duplication
            ),
            row=1, col=2
        )
    
    # Update layout
    fig.update_xaxes(title_text="Batch Size", row=1, col=1)
    fig.update_xaxes(title_text="Batch Size", row=1, col=2)
    fig.update_yaxes(title_text=f"Prefill Latency ({unit_label})", row=1, col=1)
    fig.update_yaxes(title_text=f"Decode Latency ({unit_label})", row=1, col=2)
    
    fig.update_layout(
        height=600,
        title_text="LLM Latency Predictions by Batch Size and Input Length",
        title_x=0.5,
        showlegend=True,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.05
        ),
        hovermode='closest'
    )
    
    return fig

def display_config(config):
    """Format and display the configuration in a readable way."""
    if "error" in config:
        return f"Error loading config: {config['error']}"
    
    output = "## Trained Predictor Configuration\n\n"
    
    # Display benchmark metadata if available
    metadata = config.get("metadata", {})
    if metadata:
        output += "### Benchmark Metadata\n\n"
        output += f"- **Benchmark Type**: {metadata.get('benchmark_type', 'N/A')}\n"
        output += f"- **Model Path**: {metadata.get('model_path', 'N/A')}\n"
        output += f"- **Max Batch Size**: {metadata.get('max_batch_size', 'N/A')}\n"
        output += f"- **Max Input Tokens**: {metadata.get('max_input_tokens', 'N/A')}\n"
        output += f"- **Output Length**: {metadata.get('output_len', 'N/A')}\n"
        output += f"- **Tensor Parallelism Size**: {metadata.get('tp_size', 'N/A')}\n"
        output += f"- **Pipeline Parallelism Size**: {metadata.get('pp_size', 'N/A')}\n"
        output += f"- **GPU Type**: {metadata.get('gpu_type', 'N/A')}\n"
        
        # Additional metadata if available
        if metadata.get('batch_sizes_tested'):
            output += f"- **Batch Sizes Tested**: {metadata['batch_sizes_tested']}\n"
        if metadata.get('input_lens_tested'):
            output += f"- **Input Lengths Tested**: {metadata['input_lens_tested']}\n"
        if metadata.get('num_runs'):
            output += f"- **Number of Runs**: {metadata['num_runs']}\n"
        if metadata.get('failed_configs'):
            output += f"- **Failed Configurations**: {len(metadata['failed_configs'])} configs failed\n"
        
        output += "\n"
    
    # Display model configurations
    for config_name, config_data in config.items():
        if config_name == "metadata":
            continue  # Skip metadata as we already displayed it
            
        output += f"### {config_name}\n\n"
        
        # Prefill model
        prefill = config_data.get("prefill", {})
        output += "**Prefill Model:**\n"
        output += f"- Model Type: {prefill.get('model_type', 'N/A')}\n"
        output += f"- Bias: {prefill.get('bias', 'N/A'):.8f}\n"
        output += f"- Weights: {prefill.get('weights', [])}\n\n"
        
        # Decode model
        decode = config_data.get("decode", {})
        output += "**Decode Model:**\n"
        output += f"- Model Type: {decode.get('model_type', 'N/A')}\n"
        output += f"- Bias: {decode.get('bias', 'N/A'):.8f}\n"
        output += f"- Weights: {decode.get('weights', [])}\n\n"
    
    return output

# Load the configuration
config = load_predictor_config()

# Create Gradio interface
with gr.Blocks(title="LLM Latency Predictor Viewer") as demo:
    gr.Markdown("# LLM Latency Predictor Viewer")
    gr.Markdown("Visualize trained linear regression models for prefill and decode latency prediction")
    
    # Configuration section
    gr.Markdown("## Predictor Configuration")
    config_display = gr.Markdown(display_config(config))
    
    with gr.Row():
        refresh_btn = gr.Button("Refresh Config")
        refresh_btn.click(
            fn=lambda: display_config(load_predictor_config()),
            outputs=config_display
        )
    
    # Prediction section
    gr.Markdown("## Latency Prediction")
    
    with gr.Row():
        batch_size_input = gr.Number(
            label="Batch Size", 
            value=8, 
            minimum=1, 
            maximum=64,
            step=1
        )
        input_len_input = gr.Number(
            label="Input Length", 
            value=1024, 
            minimum=1, 
            maximum=16384,
            step=1
        )
    
    with gr.Row():
        predict_btn = gr.Button("Predict Latency", variant="primary")
    
    with gr.Row():
        prefill_output = gr.Textbox(label="Prefill Latency (seconds & ms)", interactive=False)
        decode_output = gr.Textbox(label="Decode Latency (seconds & ms)", interactive=False)
    
    def predict_both(batch_size, input_len):
        current_config = load_predictor_config()
        prefill_pred = predict_latency(batch_size, input_len, current_config, "prefill")
        decode_pred = predict_latency(batch_size, input_len, current_config, "decode")
        return prefill_pred, decode_pred
    
    predict_btn.click(
        fn=predict_both,
        inputs=[batch_size_input, input_len_input],
        outputs=[prefill_output, decode_output]
    )
    
    # Visualization section
    gr.Markdown("## Latency Visualization")
    
    with gr.Row():
        max_batch_size_input = gr.Number(
            label="Max Batch Size for Plot", 
            value=64, 
            minimum=1, 
            maximum=64,
            step=1
        )
        max_input_len_input = gr.Number(
            label="Max Input Length for Plot", 
            value=32768, 
            minimum=256, 
            maximum=32768,
            step=256
        )
        unit_mode_input = gr.Dropdown(
            label="Display Units",
            choices=["seconds", "milliseconds"],
            value="seconds"
        )
    
    plot_btn = gr.Button("Generate Plots", variant="primary")
    latency_plot = gr.Plot()
    
    def update_plot(max_bs, max_il, unit_mode):
        current_config = load_predictor_config()
        return create_latency_plot(current_config, max_bs, max_il, unit_mode)
    
    plot_btn.click(
        fn=update_plot,
        inputs=[max_batch_size_input, max_input_len_input, unit_mode_input],
        outputs=latency_plot
    )
    
    # Auto-generate initial plot
    demo.load(
        fn=lambda: create_latency_plot(config, 64, 32768, "seconds"),
        outputs=latency_plot
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
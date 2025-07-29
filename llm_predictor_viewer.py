import gradio as gr
import json
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from train_utils import preprocess_input_for_prediction

def load_predictor_config(config_path="trained_predictors.json"):
    """Load the trained predictor configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        return {"error": "Config file not found"}
    except json.JSONDecodeError:
        return {"error": "Invalid JSON format"}

def predict_latency(batch_size, input_len, config, config_name, stage="prefill"):
    """Predict latency using the linear regression model."""
    if "error" in config:
        return f"Error: {config['error']}"
    
    try:
        # Get model parameters from the selected config
        if config_name not in config:
            return f"Error: Config '{config_name}' not found"
        
        model_config = config[config_name][stage]
        weights = np.array(model_config["weights"])
        bias = model_config["bias"]
        
        # Preprocess input features
        features = preprocess_input_for_prediction(batch_size, input_len, "gpu", stage)
        features_array = np.array(features)
        
        # Make prediction
        prediction = np.dot(weights, features_array) + bias
        prediction_ms = prediction * 1000
        
        # Calculate tokens/sec based on stage
        if stage == "prefill":
            tokens_per_sec = (batch_size * input_len) / prediction if prediction > 0 else 0
        else:  # decode
            tokens_per_sec = batch_size / prediction if prediction > 0 else 0
        
        return f"{prediction:.6f} seconds ({prediction_ms:.3f} ms) ({tokens_per_sec:.1f} tokens/sec)"
    except Exception as e:
        return f"Error in prediction: {str(e)}"

def create_latency_plot(config, config_name, max_batch_size=64, max_input_len=32768, unit_mode="seconds"):
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
    
    if config_name not in config:
        # Create error plot for missing config
        fig = sp.make_subplots(
            rows=1, cols=2,
            subplot_titles=("Prefill Latency", "Decode Latency"),
            horizontal_spacing=0.15
        )
        
        # Add error text annotations
        fig.add_annotation(
            text=f"Error: Config '{config_name}' not found",
            xref="x", yref="y",
            x=0.5, y=0.5,
            showarrow=False,
            row=1, col=1
        )
        fig.add_annotation(
            text=f"Error: Config '{config_name}' not found",
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
            weights = np.array(config[config_name]["prefill"]["weights"])
            bias = config[config_name]["prefill"]["bias"]
            latency = np.dot(weights, features) + bias
            latency_display = latency * unit_multiplier
            prefill_latencies.append(latency_display)
            
            # Create detailed hover text
            latency_ms = latency * 1000
            tokens_per_sec = (bs * input_len) / latency if latency > 0 else 0
            hover_texts.append(
                f"Batch Size: {bs}<br>"
                f"Input Length: {input_len}<br>"
                f"Prefill Latency: {latency:.6f}s ({latency_ms:.3f}ms) ({tokens_per_sec:.1f} tokens/sec)<br>"
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
            weights = np.array(config[config_name]["decode"]["weights"])
            bias = config[config_name]["decode"]["bias"]
            latency = np.dot(weights, features) + bias
            latency_display = latency * unit_multiplier
            decode_latencies.append(latency_display)
            
            # Create detailed hover text
            latency_ms = latency * 1000
            tokens_per_sec = bs / latency if latency > 0 else 0
            hover_texts.append(
                f"Batch Size: {bs}<br>"
                f"Input Length: {input_len}<br>"
                f"Decode Latency: {latency:.6f}s ({latency_ms:.3f}ms) ({tokens_per_sec:.1f} tokens/sec)<br>"
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

def display_config(config, config_name):
    """Format and display the configuration in a readable way."""
    if "error" in config:
        return f"Error loading config: {config['error']}"
    
    if config_name not in config:
        return f"Error: Config '{config_name}' not found"
    
    output = f"## Trained Predictor Configuration: {config_name}\n\n"
    
    config_data = config[config_name]
    
    # Display benchmark metadata if available
    metadata = config_data.get("metadata", {})
    if metadata:
        output += "### Benchmark Metadata\n\n"
        output += f"- **Benchmark Type**: {metadata.get('benchmark_type', 'N/A')}\n"
        output += f"- **Model Path**: {metadata.get('model_path', 'N/A')}\n"
        output += f"- **Max Batch Size**: {metadata.get('max_batch_size', 'N/A')}\n"
        output += f"- **Max Input Tokens**: {metadata.get('max_input_tokens', 'N/A')}\n"
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
    
    # Prefill model
    prefill = config_data.get("prefill", {})
    output += "**Prefill Model Prediction Acc:**\n"
    if 'mse' in prefill:
        output += f"- MSE: {prefill['mse']:.8f}\n"
    output += "\n"
    
    # Decode model
    decode = config_data.get("decode", {})
    output += "**Decode Model Prediction Acc:**\n"
    if 'mse' in decode:
        output += f"- MSE: {decode['mse']:.8f}\n"
    output += "\n"
    
    return output

def get_available_configs(config):
    """Get list of available config names."""
    if "error" in config:
        return []
    return [name for name in config.keys() if name != "metadata"]

# Load the configuration
config = load_predictor_config()

# Create Gradio interface
with gr.Blocks(title="LLM Latency Predictor Viewer") as demo:
    gr.Markdown("# LLM Latency Predictor Viewer")
    gr.Markdown("Visualize trained linear regression models for prefill and decode latency prediction")
    
    # Configuration selection section
    gr.Markdown("## Configuration Selection")
    with gr.Row():
        config_dropdown = gr.Dropdown(
            label="Select Configuration",
            choices=get_available_configs(config),
            value=get_available_configs(config)[0] if get_available_configs(config) else None,
            interactive=True
        )
        refresh_btn = gr.Button("Refresh Config")
    
    # Configuration display section
    gr.Markdown("## Predictor Configuration")
    config_display = gr.Markdown(
        display_config(config, config_dropdown.value) if config_dropdown.value else "No configuration selected"
    )
    
    def refresh_config():
        new_config = load_predictor_config()
        available_configs = get_available_configs(new_config)
        first_config = available_configs[0] if available_configs else None
        config_info = display_config(new_config, first_config) if first_config else "No configuration found"
        return gr.Dropdown(choices=available_configs, value=first_config), config_info
    
    def update_config_display(selected_config):
        current_config = load_predictor_config()
        return display_config(current_config, selected_config) if selected_config else "No configuration selected"
    
    refresh_btn.click(
        fn=refresh_config,
        outputs=[config_dropdown, config_display]
    )
    
    config_dropdown.change(
        fn=update_config_display,
        inputs=config_dropdown,
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
    
    def predict_both(batch_size, input_len, selected_config):
        if not selected_config:
            return "No configuration selected", "No configuration selected"
        current_config = load_predictor_config()
        prefill_pred = predict_latency(batch_size, input_len, current_config, selected_config, "prefill")
        decode_pred = predict_latency(batch_size, input_len, current_config, selected_config, "decode")
        return prefill_pred, decode_pred
    
    predict_btn.click(
        fn=predict_both,
        inputs=[batch_size_input, input_len_input, config_dropdown],
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
    
    def update_plot(max_bs, max_il, unit_mode, selected_config):
        if not selected_config:
            return create_latency_plot({"error": "No configuration selected"}, "", max_bs, max_il, unit_mode)
        current_config = load_predictor_config()
        return create_latency_plot(current_config, selected_config, max_bs, max_il, unit_mode)
    
    plot_btn.click(
        fn=update_plot,
        inputs=[max_batch_size_input, max_input_len_input, unit_mode_input, config_dropdown],
        outputs=latency_plot
    )
    
    # Auto-generate initial plot
    def initial_plot():
        initial_config = get_available_configs(config)[0] if get_available_configs(config) else ""
        return create_latency_plot(config, initial_config, 64, 32768, "seconds")
    
    demo.load(
        fn=initial_plot,
        outputs=latency_plot
    )

if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
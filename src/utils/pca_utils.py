from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch
from typing import Iterable, List, Optional, Sequence, Union, Dict, Any


def prepare_embeddings_for_pca(embeddings_list: list[np.ndarray]) -> np.ndarray:
    """
    Prepares embeddings for PCA by concatenating them.
    """
    if not embeddings_list:
        raise ValueError("Embeddings list is empty. Cannot proceed.")
    
    concatenated_embeddings = np.concatenate(embeddings_list, axis=0)
    print(f"Shape of concatenated embeddings for PCA/Dimensionality Reduction: {concatenated_embeddings.shape}")
    return concatenated_embeddings

def perform_dimensionality_reduction(embeddings: np.ndarray, method="pca", n_components=2, random_state=42, **kwargs) -> np.ndarray:
    """
    Performs dimensionality reduction (PCA by default) on the given embeddings.
    """
    print(f"\n--- Performing {method.upper()} with {n_components} components ---")
    if method.lower() == "pca":
        operator = PCA(n_components=n_components, random_state=random_state)
    # Add elif for "phate", "umap", etc. if you integrate them later
    # elif method.lower() == "phate":
    #     from phate import PHATE
    #     operator = PHATE(n_components=n_components, random_state=random_state, **kwargs)
    else:
        raise ValueError(f"Unsupported dimensionality reduction method: {method}")
        
    reduced_embeddings = operator.fit_transform(embeddings)
    print(f"Shape of {n_components}D {method.upper()} embeddings: {reduced_embeddings.shape}")
    eigenvectors = operator.components_[:n_components] if hasattr(operator, 'components_') else None
    return reduced_embeddings, eigenvectors

def create_plot_dataframe(
    reduced_embeddings: np.ndarray,
    all_labels: Optional[list[str]],
    all_positions: Optional[list[int]] | Optional[torch.Tensor], # Union type for flexibility
    words_for_df: list[str],
    additional_metadata: Optional[Dict[str, List[Any]]] = None # NEW: For arbitrary extra columns
) -> pd.DataFrame:
    """
    Creates a DataFrame for plotting from reduced embeddings and metadata.
    Detects dimensionality (2D or 3D) from reduced_embeddings shape.
    Allows inclusion of additional metadata columns.
    """
    if not all_labels: # Make sure all_labels has a default if None, e.g. for length checking
        all_labels = [""] * reduced_embeddings.shape[0] # Or handle differently based on your needs

    if reduced_embeddings.shape[0] != len(all_labels):
        raise ValueError(f"Mismatch between number of embeddings ({reduced_embeddings.shape[0]}) and number of labels ({len(all_labels)})!")
    if words_for_df and reduced_embeddings.shape[0] != len(words_for_df):
         raise ValueError(f"Mismatch between number of embeddings ({reduced_embeddings.shape[0]}) and number of words ({len(words_for_df)})!")
    if all_positions is not None:
        if isinstance(all_positions, torch.Tensor) and reduced_embeddings.shape[0] != all_positions.shape[0]:
             raise ValueError(f"Mismatch between number of embeddings ({reduced_embeddings.shape[0]}) and number of positions ({all_positions.shape[0]})!")
        elif isinstance(all_positions, (list, np.ndarray)) and reduced_embeddings.shape[0] != len(all_positions):
             raise ValueError(f"Mismatch between number of embeddings ({reduced_embeddings.shape[0]}) and number of positions ({len(all_positions)})!")


    # Ensure positions_for_df is a Python list of ints or floats
    positions_for_df: Optional[List[Union[int, float]]] = None # Initialize
    if isinstance(all_positions, torch.Tensor):
        positions_for_df = all_positions.cpu().tolist()
    elif isinstance(all_positions, np.ndarray):
        positions_for_df = all_positions.tolist()
    elif all_positions is not None: # Assuming it's already a list
        positions_for_df = all_positions
    else: # Handle case where all_positions might be None
        positions_for_df = [0] * reduced_embeddings.shape[0] # Default or raise error


    num_components = reduced_embeddings.shape[1]
    df_data: Dict[str, Any] = { # Use Dict[str, Any] for more flexibility
        'PCA1': reduced_embeddings[:, 0],
        'PCA2': reduced_embeddings[:, 1],
        'label': all_labels if all_labels is not None else [""] * reduced_embeddings.shape[0],
        'word': words_for_df if words_for_df is not None else [""] * reduced_embeddings.shape[0],
        'position': positions_for_df
    }

    if num_components >= 3:
        df_data['PCA3'] = reduced_embeddings[:, 2]
        print("Creating DataFrame for 3D plotting.")
    else:
        print("Creating DataFrame for 2D plotting.")

    # Add additional metadata if provided
    if additional_metadata:
        for key, values in additional_metadata.items():
            if len(values) != reduced_embeddings.shape[0]:
                raise ValueError(
                    f"Length of additional metadata '{key}' ({len(values)}) "
                    f"does not match number of embeddings ({reduced_embeddings.shape[0]})!"
                )
            if key in df_data:
                print(f"Warning: Additional metadata key '{key}' will overwrite an existing column in the plot DataFrame.")
            df_data[key] = values
            
    df_plot = pd.DataFrame(df_data)
    print("DataFrame for plotting created successfully:")
    print(df_plot.head())
    return df_plot

def plot_dimensionality_reduction_results(
    df_plot: pd.DataFrame,
    words_in_plot: list[str],
    output_dir: str,
    base_plot_name: str, # e.g., "Final_Token_Embeddings" or "Initial_Token_Embeddings"
    reduction_method_label: str = "PCA", # e.g., "PCA", "PHATE"
    interactive_text_column: Optional[str] = 'position', # NEW: Control for text labels, defaults to 'position'
    title: Optional[str] = None # Optional title for the plots, if None, defaults to base_plot_name
):
    """
    Generates interactive HTML and static PNG plots for 2D or 3D reduced embeddings.
    Determines dimensionality based on the presence of 'PCA3' column in df_plot.
    Allows specifying a column for interactive text labels or disabling them by setting interactive_text_column=None.
    """
    os.makedirs(output_dir, exist_ok=True)

    is_3d = 'PCA3' in df_plot.columns
    plot_dimensionality_str = "3D" if is_3d else "2D"
    
    # Common arguments for scatter functions
    scatter_args_common = {
        'color': 'position',
        'symbol': 'word',
        'hover_data': ['label', 'word', 'position'] # 'text' will be added for interactive if specified
    }
    
    scatter_args_interactive = scatter_args_common.copy()
    text_active_in_interactive_plot = False

    if interactive_text_column:
        if interactive_text_column in df_plot.columns:
            scatter_args_interactive['text'] = interactive_text_column
            text_active_in_interactive_plot = True
            print(f"Interactive plot will use '{interactive_text_column}' for text labels.")
        else:
            print(f"Warning: interactive_text_column '{interactive_text_column}' not found in DataFrame. "
                  "No text labels will be shown on the interactive plot.")
    else:
        print("No text column specified for interactive plot; text labels will be omitted.")

    # Common layout arguments
    common_layout_args = {
        'width': 1200,
        'height': 900,
        'coloraxis_colorbar_title_text': 'Position',
        'legend_title_text': 'Word'
    }

    # --- Interactive Plot ---
    interactive_title = (
        f"{plot_dimensionality_str} {reduction_method_label}: {base_plot_name} by Position (Interactive) for {'_'.join(words_in_plot)}"
    )
    
    if is_3d:
        fig_interactive = px.scatter_3d(df_plot, x='PCA1', y='PCA2', z='PCA3', **scatter_args_interactive, title=interactive_title)
        fig_interactive.update_layout(
            **common_layout_args,
            scene=dict(xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2', zaxis_title=f'{reduction_method_label}3')
        )
        # Conditional text styling
        marker_config_interactive = {'size': 5}
        trace_update_args_interactive = {'marker': marker_config_interactive}
        if text_active_in_interactive_plot:
            trace_update_args_interactive['textposition'] = 'top center'
            trace_update_args_interactive['textfont_size'] = 8
        fig_interactive.update_traces(**trace_update_args_interactive)

    else: # 2D
        fig_interactive = px.scatter(df_plot, x='PCA1', y='PCA2', **scatter_args_interactive, title=interactive_title)
        fig_interactive.update_layout(
            **common_layout_args,
            xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2',
            plot_bgcolor='rgba(240, 240, 240, 0.95)'
        )
        # Conditional text styling
        marker_config_interactive = {'size': 7}
        trace_update_args_interactive = {'marker': marker_config_interactive}
        if text_active_in_interactive_plot:
            trace_update_args_interactive['textposition'] = 'top center'
            trace_update_args_interactive['textfont_size'] = 8
        fig_interactive.update_traces(**trace_update_args_interactive)
        
        fig_interactive.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
        fig_interactive.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')

    if title == None:
        interactive_filename = f"{'_'.join(words_in_plot)}_{base_plot_name}_{reduction_method_label}_{plot_dimensionality_str}_interactive.html"
        html_path = os.path.join(output_dir, interactive_filename)
    else:
        interactive_filename = f"{title}_interactive.html"
        html_path = os.path.join(output_dir, interactive_filename)
    fig_interactive.write_html(html_path)
    print(f"Interactive {plot_dimensionality_str} {reduction_method_label} plot saved to: {html_path}")

    # --- Static Plot ---
    # Static plot usually avoids direct text labels on points for clarity, especially for 3D.
    # Hover data is still available if opened in a Plotly-aware viewer, but image export is static.
    static_title = (
        f"{plot_dimensionality_str} {reduction_method_label}: {base_plot_name} by Position (Static) for {'_'.join(words_in_plot)}"
    )
    if is_3d:
        fig_static = px.scatter_3d(df_plot, x='PCA1', y='PCA2', z='PCA3', **scatter_args_common, title=static_title)
        fig_static.update_layout(
            **common_layout_args,
            scene=dict(xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2', zaxis_title=f'{reduction_method_label}3')
        )
        fig_static.update_traces(marker=dict(size=4))
    else: # 2D
        fig_static = px.scatter(df_plot, x='PCA1', y='PCA2', **scatter_args_common, title=static_title)
        fig_static.update_layout(
            **common_layout_args,
            xaxis_title=f'{reduction_method_label}1', yaxis_title=f'{reduction_method_label}2',
            plot_bgcolor='rgba(240, 240, 240, 0.95)'
        )
        fig_static.update_traces(marker=dict(size=6))
        fig_static.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
        fig_static.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(200, 200, 200, 0.5)')
    
    static_filename_base = f"{'_'.join(words_in_plot)}_{base_plot_name}_{reduction_method_label}_{plot_dimensionality_str}_static"
    png_path = os.path.join(output_dir, f"{static_filename_base}.png")
    try:
        fig_static.write_image(png_path)
        print(f"Static {plot_dimensionality_str} {reduction_method_label} plot saved to: {png_path}")
    except ValueError as e:
        if "kaleido" in str(e).lower() or "indexOf is not a function" in str(e).lower() or "error code 525" in str(e).lower():
            print(f"Kaleido error for static {plot_dimensionality_str} {reduction_method_label} plot: {e}. Skipping PNG save.")
            print("Common fixes: ensure Kaleido is installed (`pip install -U kaleido plotly`), or simplify the figure.")
        else:
            print(f"An unexpected error occurred while writing static {plot_dimensionality_str} {reduction_method_label} plot: {e}")

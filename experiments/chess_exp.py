import sys
import os
# Add the parent directory to the system path to import modules from there
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.grammaticality_exp import generate_pca_plots_from_datasets

import torch
import random
import sys
import os
import torch
from sklearn.decomposition import PCA
import plotly.express as px
import pandas as pd
import numpy as np
import os
import torch
from sklearn.metrics import silhouette_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from tqdm.auto import tqdm

.pca_utils import (
    perform_dimensionality_reduction,
    create_plot_dataframe
)
from sonar.models.sonar_text import (
    load_sonar_tokenizer,
)
.generate_random_sequences import generate_random_sequences

# SONAR and fairseq2 imports
from sonar.models.sonar_text import load_sonar_tokenizer
# Plotting
import plotly.express as px
from src.custom_sonar_pipeline import CustomTextToEmbeddingPipeline
import numpy as np
#%%
global DEVICE, RANDOM_STATE, MODEL_NAME, OUTPUT_DIR
# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RANDOM_STATE = 42
MODEL_NAME = "text_sonar_basic_encoder"

OUTPUT_DIR = "../data/"

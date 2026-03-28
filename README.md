# Airport Link Prediction using Graph Neural Networks ✈️

## Kerollos Lowandy

**Repository:** [airport-link-prediction-gnn](https://github.com/Kerollosl/airport-link-prediction-gnn)

## 📋 Overview

This project implements Graph Neural Networks (GNN) and Classic Neural Networks to predict potential new airline routes between airports based on existing flight network data. The model analyzes airport connectivity patterns, geographical locations, and network features to identify high-probability future connections.

## 🎯 Project Goals

- Predict potential new airline routes using graph-based machine learning
- Compare GNN (GraphSAGE) vs Classic NN performance
- Visualize global airport network with geographical positioning
- Analyze network centrality metrics and flight patterns

## 🧠 Model Architectures

### 1. Graph Neural Network (GraphSAGE)
- **Implementation:** `Airline_Graph (GNN).ipynb`
- **Architecture:** GraphSAGE with 2 convolutional layers
- **Features:** Node2Vec embeddings + geographical coordinates
- **Accuracy:** ~92.5%

### 2. Classic Neural Network
- **Implementation:** `Airline_Graph (Classic NN).ipynb`
- **Architecture:** Multi-layer perceptron
- **Features:** Traditional network features

## 📊 Dataset

- **Nodes:** 3,504 airports worldwide
- **Edges:** Flight routes with distance information
- **Node Features:**
  - Airport code, city, country
  - Geographical coordinates (latitude, longitude)
  - Network centrality metrics
- **Edge Features:**
  - Flight distance in miles
  - Route frequency

**Data Files:**
- `air-routes-latest-nodes.csv` - Airport information
- `air-routes-latest-edges.csv` - Flight route connections
- `wikipedia-iso-country-codes.csv` - Country code mappings

## 🚀 Quick Start

### Prerequisites
```bash
pip install dgl node2vec networkx pandas numpy matplotlib torch scikit-learn
```

### Running the Notebooks

**Option 1: Graph Neural Network**
```bash
jupyter notebook "Airline_Graph (GNN).ipynb"
```

**Option 2: Classic Neural Network**
```bash
jupyter notebook "Airline_Graph (Classic NN).ipynb"
```

### Running Python Scripts
```bash
python airport_graph.py
```

## 📁 Repository Contents

### Notebooks
- `Airline_Graph (GNN).ipynb` - Complete GNN implementation with visualization
- `Airline_Graph (Classic NN).ipynb` - Classic NN baseline comparison

### Python Scripts
- `airport_graph.py` - Main graph construction and analysis
- `functions.py` - Utility functions for data processing and visualization

### Data Files
- `air-routes-latest-nodes.csv` - Airport nodes (3,504 airports)
- `air-routes-latest-edges.csv` - Flight route edges
- `wikipedia-iso-country-codes.csv` - Country reference data
- `potential_new.csv` - Predicted new route connections
- `pred_vs_actual.csv` - Model predictions vs actual routes

### Model Files
- `link_prediction_model.pth` - Trained GNN model weights

### Visualizations
- `Simple Graph.png` - Basic network visualization
- `Airport_Graph_with_potential_new_connections.png` - Predicted routes overlay
- `Emphasize_Upper_Graph_Top_15_15_More_Degree_8500_More_Edges.png` - Major hub analysis
- `Emphasize_Less_Graph_Bottom_15_50_Less_Degree_50_Less_Edges.png` - Minor airport analysis
- `Confusion_Matrix.png` - Model performance metrics

## 🔍 Key Features

### Network Analysis
- **Degree Centrality:** Identifies major airline hubs
- **Betweenness Centrality:** Finds critical connection points
- **Closeness Centrality:** Measures airport accessibility
- **Distance Analysis:** Longest/shortest flights, average distances

### Visualization
- Geographical network plots with real coordinates
- Interactive filtering by degree threshold
- Edge highlighting by distance
- Country-based labeling

### Model Features
- Node2Vec embeddings for structural features
- Geographical coordinate integration
- Link prediction with 92%+ accuracy
- Train/test split with negative sampling

## 📈 Results

- **Test Accuracy:** 92.46%
- **Precision:** 92.46%
- **Recall:** 92.46%
- **F1-Score:** 92.46%

## 🛠️ Technical Stack

- **Deep Learning:** PyTorch, DGL (Deep Graph Library)
- **Graph Processing:** NetworkX, Node2Vec
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib
- **ML Tools:** Scikit-learn

## 📝 Usage Examples

### Visualize Airport Network
```python
plot_detailed_graph(G, country_labels, 
                   emphasize_greater=True, 
                   degree_threshold=15,
                   edge_highlight=8500)
```

### Get Airport Metrics
```python
print_metrics(reverse=True, count=10,
             node=True, 
             betweenness=True,
             closeness=True)
```

### Train GNN Model
```python
model = GraphSAGE(in_feats=130, h_feats=16)
pred = MLPPredictor(16)
# Training loop with 500 epochs
```

## 🎓 Applications

- Airline route planning and optimization
- Airport infrastructure investment decisions
- Travel demand forecasting
- Network resilience analysis
- Hub identification for new airlines

## 👤 Author

**Kerollos Lowandy**
- GitHub: [@Kerollosl](https://github.com/Kerollosl)
- Email: klowandy@gmail.com

## 📄 License

This project is available for educational and research purposes.

---

**Last Updated:** March 27, 2026

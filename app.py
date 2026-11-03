
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

st.set_page_config(
    page_title="Patient Genetic Phenotype Segmentation",
    layout="wide"
)

st.title("Patient Genetic Phenotype Segmentation System")

st.subheader("K-Means Clustering Dashboard")

# Load files
df = pd.read_csv('patient_clusters.csv')

pca_df = pd.read_csv('pca_visualization.csv')

st.sidebar.header("Dashboard Settings")

st.sidebar.write(
    "Patient Genetic Phenotype Segmentation"
)

# Number of patients
st.metric(
    "Total Number of Patients",
    len(df)
)

# Elbow curve
st.header("Elbow Curve")

inertia_df = pd.DataFrame({
    'K': list(range(2, 11)),
    'Inertia': inertia_values
})

st.line_chart(
    inertia_df.set_index('K')
)

# Silhouette score
st.header("Silhouette Score Comparison")

silhouette_df = pd.DataFrame({
    'K': list(range(2, 11)),
    'Silhouette Score': silhouette_values
})

st.line_chart(
    silhouette_df.set_index('K')
)

# Cluster visualization
st.header("Patient Cluster Visualization")

fig, ax = plt.subplots(figsize=(10, 6))

for cluster in sorted(pca_df['Cluster'].unique()):

    cluster_data = pca_df[
        pca_df['Cluster'] == cluster
    ]

    ax.scatter(
        cluster_data['PCA 1'],
        cluster_data['PCA 2'],
        label=f'Cluster {cluster}'
    )

ax.set_xlabel("Principal Component 1")
ax.set_ylabel("Principal Component 2")
ax.legend()

st.pyplot(fig)

# Cluster counts
st.header("Number of Patients in Each Cluster")

cluster_counts = df['Cluster'].value_counts().sort_index()

st.bar_chart(cluster_counts)

# Numerical characteristics
st.header("Average Numerical Feature Values")

numerical_features = df.select_dtypes(
    include=['int64', 'float64']
).columns.tolist()

numerical_features = [
    col for col in numerical_features
    if col != 'Cluster'
]

characteristics = df.groupby(
    'Cluster'
)[numerical_features].mean()

st.dataframe(characteristics)

# Categorical characteristics
st.header("Common Categorical Characteristics")

categorical_features = df.select_dtypes(
    include=['object', 'category', 'bool']
).columns.tolist()

for cluster in sorted(df['Cluster'].unique()):

    st.subheader(f"Cluster {cluster}")

    cluster_data = df[
        df['Cluster'] == cluster
    ]

    common_values = {}

    for col in categorical_features:

        mode_values = cluster_data[col].mode()

        if not mode_values.empty:

            common_values[col] = mode_values.iloc[0]

    st.write(common_values)

st.success(
    "Clusters show similarity in dataset features."
)

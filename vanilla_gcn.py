
import os
import random

import torch
import torch.nn.functional as F
import networkx as nx
import matplotlib.pyplot as plt

from torch_geometric.datasets import KarateClub
from torch_geometric.nn import GCNConv
from sklearn.manifold import TSNE


# 1. Reproducibility
random.seed(42)
torch.manual_seed(42)

os.makedirs("outputs", exist_ok=True)

# 2. Load dataset
dataset = KarateClub()
data = dataset[0]

print("Dataset: Zachary Karate Club")
print("Number of nodes:", data.num_nodes)
print("Number of edges:", data.edge_index.size(1) // 2)
print("Number of node features:", dataset.num_features)
print("Number of classes:", dataset.num_classes)
print("Training nodes:", int(data.train_mask.sum()))


# 3. Visualize the graph
G = nx.karate_club_graph()

plt.figure(figsize=(9, 7))
pos = nx.spring_layout(G, seed=42)

nx.draw_networkx(
    G,
    pos=pos,
    node_color=data.y.numpy(),
    cmap=plt.cm.Set2,
    node_size=500,
    font_size=9,
    edge_color="gray",
    with_labels=True
)

plt.title("Zachary Karate Club Network")
plt.axis("off")
plt.tight_layout()
plt.savefig("outputs/karate_network.png", dpi=200)
plt.close()


# 4. Define Vanilla GCN
class VanillaGCN(torch.nn.Module):
    def __init__(self, input_features, hidden_features, num_classes):
        super().__init__()

        self.conv1 = GCNConv(input_features, hidden_features)
        self.conv2 = GCNConv(hidden_features, num_classes)

    def forward(self, data):
        x = self.conv1(data.x, data.edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, data.edge_index)
        return x


model = VanillaGCN(
    dataset.num_features,
    16,
    dataset.num_classes
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss()

# 5. Train using the labeled training nodes
loss_history = []

for epoch in range(1, 201):
    model.train()
    optimizer.zero_grad()

    output = model(data)

    loss = criterion(
        output[data.train_mask],
        data.y[data.train_mask]
    )

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Loss: {loss.item():.4f}")

print("\nTraining completed!")


# 6. Evaluate predictions
model.eval()

with torch.no_grad():
    logits = model(data)
    predictions = logits.argmax(dim=1)

correct_train = (
    predictions[data.train_mask] == data.y[data.train_mask]
).sum().item()

train_accuracy = correct_train / int(data.train_mask.sum())

# This is accuracy over all labeled dataset nodes, not held-out test accuracy.
overall_accuracy = (
    predictions == data.y
).float().mean().item()

print(f"Training accuracy: {train_accuracy:.2%}")
print(f"Overall dataset accuracy: {overall_accuracy:.2%}")
print("Note: Overall accuracy is not held-out test accuracy.")


# 7. Save loss curve
plt.figure(figsize=(8, 5))
plt.plot(range(1, 201), loss_history)
plt.xlabel("Epoch")
plt.ylabel("Training Loss")
plt.title("Vanilla GCN Training Loss")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("outputs/training_loss.png", dpi=200)
plt.close()


# 8. Extract first-layer node embeddings
with torch.no_grad():
    embeddings = model.conv1(data.x, data.edge_index)
    embeddings = F.relu(embeddings)

print("Embedding shape:", tuple(embeddings.shape))
print("First five node embeddings:\n", embeddings[:5])


# 9. Visualize embeddings using t-SNE
embedding_array = embeddings.cpu().numpy()

tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=10,
    init="pca",
    learning_rate="auto"
)

embedding_2d = tsne.fit_transform(embedding_array)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    embedding_2d[:, 0],
    embedding_2d[:, 1],
    c=data.y.numpy(),
    cmap="Set2",
    s=100
)

plt.title("Vanilla GCN Node Embeddings (t-SNE)")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.colorbar(scatter, label="Class")
plt.tight_layout()
plt.savefig("outputs/embedding_output.png", dpi=200)
plt.close()


# 10. Save results to a text file
with open("outputs/output.txt", "w", encoding="utf-8") as file:
    file.write("Vanilla GCN - Karate Club Dataset\n")
    file.write(f"Nodes: {data.num_nodes}\n")
    file.write(f"Edges: {data.edge_index.size(1) // 2}\n")
    file.write(f"Training accuracy: {train_accuracy:.2%}\n")
    file.write(f"Overall dataset accuracy: {overall_accuracy:.2%}\n")
    file.write("Overall accuracy is not held-out test accuracy.\n")
    file.write(f"Embedding shape: {tuple(embeddings.shape)}\n")
    file.write(f"Final training loss: {loss_history[-1]:.4f}\n")

print("\nAll outputs saved in the outputs/ folder.")

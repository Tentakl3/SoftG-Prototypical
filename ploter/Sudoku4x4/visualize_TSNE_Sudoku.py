import torch.nn.functional as F

import matplotlib.pyplot as plt
from matplotlib import cm
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import torch
import ltn

import numpy as np

def visualize_tsne_sudoku(loader, results, num_classes, class_names, epoch=1):
    all_labels = []
    all_embeddings = []

    embedding = results['embedding']
    prototypes = results['prototypes']

    # ---- Compute embeddings ----

    with torch.no_grad():

      for batch_idx, (board_images, board_labels, digit_labels, sample_idx) in enumerate(loader):
          with torch.no_grad():
              board_images = board_images.to(ltn.device)
              board_labels = board_labels.to(ltn.device)
              digit_labels = digit_labels.to(ltn.device)

              B, N, _ = digit_labels.shape

              board_images_reshape = board_images.reshape(B*N*N, 1, 28, 28)
              z_digits = embedding(board_images_reshape)
              z_digits = F.normalize(z_digits, dim=-1).cpu().detach().numpy()
              z_digits_reshape = z_digits.reshape(B*N*N,-1)

              true_digits = (digit_labels-1).reshape(B*N*N)
              all_embeddings.append(z_digits_reshape)
              all_labels.append(true_digits)

    embeddings = np.concatenate(all_embeddings, axis=0)
    labels = torch.cat(all_labels).cpu()

    combined = np.vstack([embeddings, prototypes])

    transform = TSNE(n_components=2, random_state=42)
    #transform = PCA(n_components=2)
    embed = transform.fit_transform(combined)

    n_data = len(embeddings)
    n_fine = len(prototypes)

    feat_2d = embed[:n_data]
    fine_proto_2d = embed[n_data:n_data+n_fine]

    cmap = plt.cm.get_cmap('gist_ncar', num_classes)  # 或'nipy_spectral'
    fine_colors = cmap(labels.cpu().numpy() % num_classes)

    plt.figure(figsize=(20, 8), dpi=150)

    plt.subplot(121)
    plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=fine_colors, alpha=0.7, s=20,
                edgecolors='black', linewidth=0.3)

    for i in range(n_fine):
        plt.scatter(fine_proto_2d[i, 0], fine_proto_2d[i, 1],
                    marker='*', s=200, color=cmap(i % num_classes),
                    edgecolors='black', linewidth=1, zorder=3)

        plt.text(
            fine_proto_2d[i, 0] + 2,  # slight offset for readability
            fine_proto_2d[i, 1] + 2,
            f'{class_names[i]}', fontsize=10, fontweight='bold',
            color='black', ha='left', va='center',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1)
        )

    plt.title(f'Prototypes and Features')

    """
    plt.subplot(122)
    plt.scatter(feat_2d[:, 0], feat_2d[:, 1], c=fine_colors, alpha=0.7, s=20,
                edgecolors='w', linewidth=0.3)

    coarse_cmap = plt.cm.get_cmap('tab20', len(neg_proto_2d))
    for i in range(len(neg_proto_2d)):
        plt.scatter(neg_proto_2d[i, 0], neg_proto_2d[i, 1],
                    marker='o', s=200, color=coarse_cmap(i),
                    edgecolors='black', linewidth=1, zorder=3)
    plt.title(f'Higher-Level Prototypes and Features')
    """
    plt.tight_layout()
    #plt.savefig(f'high_color_epoch{epoch}.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
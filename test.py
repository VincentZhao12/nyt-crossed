from sklearn.metrics.pairwise import pairwise_distances
from collections import defaultdict
import numpy as np

def find_closest_group(labels, vectors):
  """
  Finds the group of 4 vectors with the smallest average distance between each other,
  excluding pairs with the same prefix.

  Args:
    labels: A list of string labels for each vector.
    vectors: A 2D numpy array where each row is a vector.

  Returns:
    A tuple containing:
      - A list of the 4 closest vector labels.
      - The average distance between the 4 vectors.
  """

  # Calculate pairwise distances between all vectors
  distances = pairwise_distances(vectors)

  # Convert distances to similarities for efficiency
  similarities = 1 - distances
  print(distances)

  # Group vectors by prefix
  grouped_vectors = defaultdict(list)
  for label, vector in zip(labels, vectors):
    prefix, _ = label.split("_")
    grouped_vectors[prefix].append((label, vector))

  # Find the highest similarity pairs excluding same prefix
  top_pairs = []
  for prefix, group in grouped_vectors.items():
    if len(group) > 1:
      # Calculate similarities within the group
      # print(similarities)
      group_similarities = similarities[list(zip(*group))[0]]
      group_mask = np.tril(np.ones(group_similarities.shape)).astype(bool)
      group_similarities[~group_mask] = 0

      # Find top (n-1) similarities within the group, excluding self-comparisons
      top_group_pairs = np.unravel_index(np.argsort(group_similarities, axis=None)[-n+1:], group_similarities.shape)
      top_pairs.extend([(group[i][0], group[j][0]) for i, _ in top_group_pairs for j in range(1, n)])

  # Extract corresponding labels and vectors
  top_labels = [pair[0] for pair in top_pairs]
  top_vectors = vectors[[label_index for label, _ in top_pairs], :]

  # Calculate the average distance between the top vectors
  avg_distance = distances[[label_index for label, _ in top_pairs] , [pair[1] for pair in top_pairs]].mean()

  return top_labels, avg_distance


# Example usage
labels = ["a_1", "a_2", "b_1", "b_2", "c_1"]
vectors = np.random.rand(5, 10)  # Replace with your actual vectors

closest_group, avg_distance = find_closest_group(labels, vectors)

print(f"Closest group of labels: {closest_group}")
print(f"Average distance between them: {avg_distance}")

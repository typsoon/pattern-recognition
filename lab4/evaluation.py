"""Fixed k-NN evaluation, baselines, and result reporting for Lab 4."""


import torch
from torch.nn import functional as F

from common import device, normalize
from network import TinyViT


@torch.no_grad()  # Evaluation features do not require gradients or a training graph.
def features(backbone, images):
    # returns: (N, dim) L2-normalized features of the frozen network
    with torch.autocast("cuda", dtype=torch.float16):  # Reduce GPU work during feature extraction.
        # Unit-length features make their dot product equal to cosine similarity.
        return F.normalize(
            torch.cat(
                [
                    # Normalize pixels, extract a manageable batch, and retain float32 features.
                    backbone(normalize(images[i : i + 1024])).float()
                    for i in range(0, len(images), 1024)
                ]
            ),
            dim=-1,
        )


def weighted_knn(bank, bank_labels, queries, query_labels, k=20, temperature=0.07):
    # returns: top-1 accuracy of a similarity-weighted vote of the k nearest stored features
    correct = 0
    for i in range(0, len(queries), 1024):  # Batch queries to limit the similarity matrix size.
        # Compare each query with every bank feature and keep its k closest neighbours.
        similarities, neighbours = (queries[i : i + 1024] @ bank.T).topk(k, dim=1)
        # Accumulate exp(similarity / temperature) into each neighbour's class bin.
        votes = torch.zeros(len(similarities), 10, device=device).scatter_add_(
            1, bank_labels[neighbours], (similarities / temperature).exp()
        )
        correct += (votes.argmax(1) == query_labels[i : i + 1024]).sum().item()  # Count winning labels.
    return correct / len(queries)  # Convert the total count into top-1 accuracy.


def knn_eval(backbone, train_images, train_labels, test_images, test_labels):
    # The fixed evaluation for task 4. Do not change it; report exactly what it returns.
    # backbone - a network taking normalized images to (batch, dim) features
    # returns: top-1 accuracy on the CIFAR-10 test set
    was_training = backbone.training  # Preserve the mode used by the calling training loop.
    backbone.eval()  # Evaluate with training-specific layers disabled.
    # Use training images as the labelled bank and test images only as queries.
    accuracy = weighted_knn(
        features(backbone, train_images),
        train_labels,
        features(backbone, test_images),
        test_labels,
    )
    backbone.train(was_training)  # Resume the caller's original model mode.
    return accuracy


def val_knn(backbone, train_images, train_labels):
    # The same as knn_eval, on the validation split. Use this one for every choice you tune.
    # returns: top-1 accuracy on the last 5,000 training images, with the other 45,000 as neighbours
    was_training = backbone.training  # Save the mode so validation does not alter training.
    backbone.eval()
    all_features = features(backbone, train_images)  # Extract the bank and queries in one pass.
    # Use disjoint parts of the training split, keeping the test set out of tuning.
    accuracy = weighted_knn(
        all_features[:45000],
        train_labels[:45000],
        all_features[45000:],
        train_labels[45000:],
    )
    backbone.train(was_training)  # Restore the mode after validation.
    return accuracy


def present_baselines(train_images, train_labels, test_images, test_labels):
    """Report the parameter count and the two prescribed baselines."""
    print("parameters:", sum(p.numel() for p in TinyViT().parameters()))
    # Flatten standardized pixels and give each image unit length for a cosine k-NN baseline.
    pixels = lambda x: F.normalize(normalize(x).flatten(1), dim=-1)  # noqa: E731
    print(
        "raw pixels, k-NN accuracy:",
        weighted_knn(pixels(train_images), train_labels, pixels(test_images), test_labels),
    )
    torch.manual_seed(0)  # Make the untrained-network baseline reproducible.
    print("untrained network, k-NN accuracy:", knn_eval(TinyViT().to(device), train_images, train_labels, test_images, test_labels))


def present_the_outcome(full, evaluate):
    # Compare student and EMA-teacher features with the fixed test-set evaluator.
    print(
        "student k-NN:",
        evaluate(full["student"].backbone),
        "| teacher k-NN:",
        evaluate(full["teacher"].backbone),
    )  # type: ignore
    # Convert PyTorch's process-wide peak allocated bytes into GiB for reporting.
    print("peak GPU memory, GB:", round(torch.cuda.max_memory_allocated() / 2**30, 1))

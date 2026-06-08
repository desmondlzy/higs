import pytest
import torch

# from gsplat._helper import load_test_data
# device = torch.device("cuda:0")

from heatgaussian.build_csr_matrix import TripletCSR, build_csr_matrix_add_row, build_csr_matrix_init

def test_build_csr_matrix():
	triplet = build_csr_matrix_init("cpu")
	triplet = build_csr_matrix_add_row(
		torch.tensor([0, 1, 2]),
		torch.tensor([1.0, 2.0, 3.0]),
		triplet,
	)
	triplet = build_csr_matrix_add_row(
		torch.tensor([0, 1, 2]),
		torch.tensor([1.0, 2.0, 3.0]),
		triplet,
	)

	wikipedia_example = torch.tensor([
		[10, 20,  0,  0,  0,  0],
		[ 0, 30,  0, 40,  0,  0],
		[ 0,  0, 50, 60, 70,  0],
		[ 0,  0,  0,  0,  0, 80],
	])

	triplet = build_csr_matrix_init("cpu")
	for i, row in enumerate(wikipedia_example):
		nonzero_indices = torch.nonzero(row).flatten()
		triplet = build_csr_matrix_add_row(
			nonzero_indices,
			row[nonzero_indices],
			triplet,
		)

	wiki_csr = torch.sparse_csr_tensor(
		triplet.crow_indices,
		triplet.col_indices,
		triplet.values,
		(4, 6),
	)

	print(triplet.values, triplet.col_indices, triplet.crow_indices)
	print(wiki_csr)
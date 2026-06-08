import torch

def sparse_tensor_memory_size(sparse_tensor: torch.Tensor) -> int:
	if sparse_tensor.is_sparse_csr:
		return (
			sparse_tensor.crow_indices().element_size() * sparse_tensor.crow_indices().nelement() + 
			sparse_tensor.col_indices().element_size() * sparse_tensor.col_indices().nelement() + 
			sparse_tensor.values().element_size() * sparse_tensor.values().nelement()) 
	elif sparse_tensor.is_sparse: # coo
		return (
			sparse_tensor.indices().element_size() * sparse_tensor.indices().nelement() + 
			sparse_tensor.values().element_size() * sparse_tensor.values().nelement())
	else:
		raise ValueError(f"only support csr/coo tensor now, but got {sparse_tensor.layout}")

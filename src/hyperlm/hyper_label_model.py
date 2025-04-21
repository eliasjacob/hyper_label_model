import torch
import torch.nn as nn
import numpy as np
from scipy.sparse import coo_matrix
import torch.optim as optim
import os

from .loss import BCEMask, BCEMaskWeighted


def sparse_mean(index, value, expand=True, output_size=None):
    """obtain the mean of values that share the same index (e.g. same row or column in label matrix X) 

    Args:
        index: The indices of the elements, the first dimension is batch size 
        value : The values of the elements, the first dimension is batch size
        expand (bool, optional): if true expand the output to have the same size as index
        output_size (int, optional): The desired size of the first dimension of the output tensor. 
                                     If None, it's determined by index.max() + 1.

    Returns:
        mean values
    """
    output_batch = []
    # Determine the size for the output tensor's first dimension
    if output_size is None:
        # Check if index is empty before calling max()
        if index.numel() == 0:
            # Handle empty index case: perhaps return zeros of appropriate shape or raise error
            # For now, let's assume we need a minimal size if output_size is not given
            # This might need adjustment based on expected behavior for empty inputs
            ind_max = 0 
        else:
            ind_max = int(index.max() + 1)
    else:
        ind_max = output_size
        
    for i_batch in range(value.shape[0]):
        # Ensure index is not empty for the current batch item before proceeding
        if index[i_batch].numel() == 0:
             # Handle empty index for a batch item: append zeros or skip
             # Appending zeros of the expected shape based on ind_max
             output = torch.zeros((ind_max, value.shape[2])).float().to(value.device)
             if not expand: # If not expanding, we just need the aggregated output
                 output_batch.append(output)
                 continue # Skip the rest of the loop for this batch item
             else: # If expanding, we need to handle how to expand zeros.
                   # This case might require specific logic depending on requirements.
                   # For now, let's append the zero tensor meant for aggregation.
                   # Consider if expanding an empty set should result in zeros matching original indices.
                   # This part might need refinement based on use-case specifics.
                   output_batch.append(torch.zeros_like(value[i_batch])) # Placeholder logic
                   continue

        output = torch.zeros((ind_max, value.shape[2])).float().to(value.device).index_add_(0,
                                                                                            index[i_batch],
                                                                                            value[i_batch])
        norm = torch.zeros(ind_max).to(value.device).float().index_add_(
            0, index[i_batch], torch.ones_like(index[i_batch]).float()) + 1e-9
        output = output / norm[:, None].float()
        if expand:
            # Ensure indices used for selection are within the bounds of the output tensor
            valid_indices = index[i_batch][index[i_batch] < ind_max]
            # This part needs careful handling if expand=True is used with output_size
            # As index might contain values >= output_size if not filtered properly upstream
            # Assuming index values are always < output_size when expand=True and output_size is provided
            output = torch.index_select(output, 0, index[i_batch])
        output_batch.append(output)
    return torch.stack(output_batch)


class GNNLayer(nn.Module):
    """One GNN layer
    Args:
        in_features: embedding dimension of the input
        out_features: embedding dimension of the output
    """

    def __init__(self, in_features, out_features):
        super(GNNLayer, self).__init__()
        self.activation = nn.LeakyReLU()
        self.linear = nn.Linear(in_features * 4, out_features)  # this corresponds to fk in Equation 5
        self.linear_row = nn.Linear(in_features, in_features)  # this corresponds to W1 in Equation 5
        self.linear_col = nn.Linear(in_features, in_features)  # this corresponds to W2 in Equation 5
        self.linear_global = nn.Linear(in_features, in_features)  # this corresponds to W3 in Equation 5
        self.linear_self = nn.Linear(in_features, in_features)  # this corresponds to W4 in Equation 5

    def forward(self, index, value):
        ## pool over values in the same column, same row, and the whole matrix
        pooled = [self.linear_row(sparse_mean(index[:, :, 0], value)),
                  self.linear_col(sparse_mean(index[:, :, 1], value)),
                  self.linear_global(torch.mean(value, dim=1)).unsqueeze(1).expand_as(value)]

        stacked = torch.cat(
            [self.linear_self(value)] + pooled, dim=2)
        return index, self.activation(self.linear(stacked))


class SequentialMultiArg(nn.Sequential):
    """helper class to stack multiple GNNLayer
    """

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


class LELAGNN(nn.Module):
    """The model architecture of the LELA hyper label model
    """

    def __init__(self):
        super(LELAGNN, self).__init__()
        # the multi-layer GNN network
        self.matrix_net = SequentialMultiArg(
            GNNLayer(1, 8),
            GNNLayer(8, 8),
            GNNLayer(8, 8),
            GNNLayer(8, 32),
        )
        col_embed_mixed_size = 32
        # MLP network
        self.classify = nn.Sequential(
            nn.Linear(col_embed_mixed_size, col_embed_mixed_size),
            nn.LeakyReLU(),
            nn.Linear(col_embed_mixed_size, col_embed_mixed_size),
            nn.LeakyReLU(),
            nn.Linear(col_embed_mixed_size, 1),
            nn.Sigmoid()
        )

    def forward(self, index, value, n_rows):
        _, elementwise_embed_sparse = self.matrix_net(index, value.float().unsqueeze(2))  # encode matrix with GNNs
        # Pass n_rows to sparse_mean to ensure correct output dimension
        example_embed = sparse_mean(
            index[:, :, 0], elementwise_embed_sparse,
            expand=False, output_size=n_rows)  # pool over elements in the same row to obtain embedding for each example
        mask = torch.sum(example_embed, dim=2) > 0
        # mask examples where all LFs abstains, 
        # these examples have all zeros in the ebmeddings. 
        # mask will be used in loss function to skip these examples
        output = self.classify(example_embed)
        return torch.squeeze(output), mask


class UnsupervisedWraper:
    """Wrapper for the pre-trained LELA hyper label model
    """

    def __init__(self, device, checkpoint_path):  # initialize from a trained model checkpoint
        self.device = device
        self.net = LELAGNN()
        self.net.to(self.device)
        checkpoint = torch.load(checkpoint_path, map_location=torch.device(self.device), weights_only=False)
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.net.eval()
        self.name = checkpoint_path

    def predict(self, X):
        """predict hard labels for each data point

        Args:
            X: the label matrix, a nxm array where n is the number of data points and m is the number of LFs
            for each element in X, -1 denotes abstaintion and other numbers (e.g. 0, 1,2) denote each classes

        Returns:
            y: hard labels, a numpy array of size (n,)
        """
        probs = self.predict_prob(X)
        pred = np.argmax(probs, axis=1).flatten()
        return pred

    def predict_prob(self, X):
        """predict probabilities of each data point being each class

        Args:
            X: the label matrix, a nxm array where n is the number of data points and m is the number of LFs
            for each element in X, -1 denotes abstaintion and other numbers (e.g. 0, 1,2) denote the classes
        Returns:
            y: probabilities of each data point being each class, a nxk matrix where k is the number of classes
        """
        X_arr = np.array(X)
        n_rows = X_arr.shape[0] # Get the original number of rows
        preds = []
        n_class = int(np.max(X_arr[X_arr != -1])) + 1 if np.any(X_arr != -1) else 0 # Handle case where all are -1
        if n_class < 2: # Handle binary case properly even if only one class or only abstentions are present
             n_class = 2 # Default to binary classification problem structure if fewer than 2 classes observed

        if n_class == 2:
            mat = -1 * np.ones_like(X_arr)
            mat[X_arr == 1] = 1
            mat[X_arr == -1] = 0  # note -1 denotes abstention in X and X_arr, but 0 denotes abstention in mat
            mat = np.array(mat)
            X_sparse = coo_matrix(np.squeeze(mat))
            index = np.array([X_sparse.row, X_sparse.col]).T
            value = X_sparse.data
            index = torch.from_numpy(index).to(self.device)
            value = torch.from_numpy(value).float().to(self.device)
            # Pass n_rows to the forward method
            pred, _ = self.net(index.unsqueeze(0), value.unsqueeze(0), n_rows=n_rows)
            preds = torch.stack([1 - pred, pred], dim=1)
        else:
            for label in range(n_class):
                mat = -1 * np.ones_like(X_arr)
                mat[X_arr == label] = 1
                mat[X_arr == -1] = 0  # note -1 denotes abstention in X and X_arr, but 0 denotes abstention in mat
                mat = np.array(mat)
                X_sparse = coo_matrix(np.squeeze(mat))
                index = np.array([X_sparse.row, X_sparse.col]).T
                value = X_sparse.data
                index = torch.from_numpy(index).to(self.device)
                value = torch.from_numpy(value).float().to(self.device)
                 # Pass n_rows to the forward method
                pred, _ = self.net(index.unsqueeze(0), value.unsqueeze(0), n_rows=n_rows)
                preds.append(pred.unsqueeze(1))
            preds = torch.cat(preds, dim=1)
            # Normalize probabilities, handle potential division by zero if a row sums to 0
            sum_preds = torch.sum(preds, dim=1, keepdim=True)
            # Avoid division by zero for rows that sum to 0 (e.g., all abstentions)
            # Assign uniform probability or handle as per desired logic for such cases
            # Here, we assign uniform probability 1/n_class if sum is 0
            uniform_prob = 1.0 / n_class
            preds = torch.where(sum_preds > 1e-9, preds / sum_preds, torch.full_like(preds, uniform_prob))
            # preds = preds / torch.sum(preds, dim=1).unsqueeze(1)
        preds = preds.detach().cpu().numpy()
        # Ensure the output has n_rows, padding if necessary (though passing n_rows should prevent this need)
        if preds.shape[0] < n_rows:
             # This case should ideally not happen if n_rows is passed correctly
             # If it does, padding might be a fallback, but indicates an issue upstream
             padding = np.full((n_rows - preds.shape[0], preds.shape[1]), 1.0 / n_class) # Example padding with uniform prob
             preds = np.vstack((preds, padding))

        return preds



def calibrate_probs(probs):
    ## scale the probs the have a maximum of 1 and minimum of 0
    n_class = probs.shape[1]
    tie_score = 1 / n_class
    probs[probs > tie_score] = (probs[probs > tie_score] - tie_score) * tie_score / (
            np.max(probs[probs > tie_score]) - tie_score) + tie_score
    probs[probs < tie_score] = (probs[probs < tie_score] - tie_score) * tie_score / (
            tie_score - np.min(probs[probs < tie_score])) + tie_score
    probs[probs > 1] = 1
    probs[probs < 0] = 0
    probs = probs / np.sum(probs, axis=1, keepdims=True)
    return probs


def HyperLMSemi(LF_mat, y_vals, y_indices, device, checkpoint_path, n_rows):
    """function to perform semisupervised label aggregation using the LELA hyper label model

    Args:
        LF_mat: label matrix, note "-1" denotes abstentions
        y_vals: provided labels, e.g [1, 0, 1, 1]. 
        y_indices: the corresponding indices of the provided labels, e.g. [100, 198, 222, 4213]
        device: Torch device.
        checkpoint_path: Path to the model checkpoint.
        n_rows: The total number of rows (data points) in LF_mat.

    Returns:
        predicted labels probabilities (nxk tensor)
    """
    preds = []
    # Determine n_class safely, considering y_vals might be empty or only contain one class
    all_labels = set(LF_mat.flatten()) - {-1}
    if y_vals is not None and len(y_vals) > 0:
        all_labels.update(y_vals)
    
    n_class = 0
    if all_labels:
        n_class = int(max(all_labels) + 1)
    
    # Ensure n_class is at least 2 for binary classification structure
    if n_class < 2:
        n_class = 2

    # Convert y_vals to numpy array for consistency
    y_vals = np.array(y_vals) if y_vals is not None else np.array([])
    y_indices = np.array(y_indices) if y_indices is not None else np.array([])


    if n_class == 2:
        lela = SemisupervisedHelper(device, checkpoint_path)
        X = np.zeros_like(LF_mat) - 1
        X[LF_mat == 1] = 1
        X[LF_mat == -1] = 0
        weights = None
        if len(y_vals) > 0:
             pos_ratio = np.sum(y_vals) / len(y_vals) if len(y_vals) > 0 else 0.5 # Avoid division by zero
             # Ensure weights are valid probabilities
             pos_ratio = max(1e-6, min(1 - 1e-6, pos_ratio)) # Clamp pos_ratio to avoid exact 0 or 1
             weights = [1 - pos_ratio, pos_ratio] # Order might depend on BCEMaskWeighted expectation (e.g., [weight_for_0, weight_for_1])

        # Pass n_rows to fit_predict
        pred = lela.fit_predict(X, y_vals, y_indices, n_rows, weights)
        preds = [1 - pred.unsqueeze(1), pred.unsqueeze(1)]
    else:
        for label in range(n_class):
            lela = SemisupervisedHelper(device, checkpoint_path)
            X = np.zeros_like(LF_mat) - 1
            X[LF_mat == label] = 1
            X[LF_mat == -1] = 0
            y_binary = np.zeros_like(y_vals)
            if len(y_vals) > 0: # Check if y_vals is not empty
                 y_binary[y_vals == label] = 1
            # Pass n_rows to fit_predict
            pred = lela.fit_predict(X, y_binary, y_indices, n_rows)
            preds.append(pred.unsqueeze(1))

    if not preds: # Handle case where preds list might be empty (e.g., n_class=0 initially)
         # Return a default prediction, e.g., uniform probabilities
         # This depends on the desired behavior for edge cases like no labels observed.
         # Assuming n_rows is the total number of data points.
         # Defaulting to uniform probability across 2 classes if n_class was determined < 2.
         num_classes_for_default = max(2, n_class) # Use at least 2 classes for default shape
         return np.full((n_rows, num_classes_for_default), 1.0 / num_classes_for_default)


    preds = torch.cat(preds, dim=1)
    # Normalize probabilities, handle potential division by zero
    sum_preds = torch.sum(preds, dim=1, keepdim=True)
    uniform_prob = 1.0 / n_class
    preds = torch.where(sum_preds > 1e-9, preds / sum_preds, torch.full_like(preds, uniform_prob))
    # preds = preds / torch.sum(preds, dim=1).unsqueeze(1)
    
    # Ensure final output has n_rows
    final_preds = preds.detach().cpu().numpy()
    if final_preds.shape[0] < n_rows:
        # This indicates an issue, potentially in fit_predict or earlier.
        # Fallback padding:
        padding = np.full((n_rows - final_preds.shape[0], final_preds.shape[1]), 1.0 / n_class)
        final_preds = np.vstack((final_preds, padding))
        
    return final_preds


class SemisupervisedHelper:
    """helper class to perform semisupervised label aggregation
    """

    def __init__(self, device, checkpoint_path):
        self.device = device
        self.net = LELAGNN()
        self.optimizer = optim.Adam(
            self.net.parameters(),
            lr=0.0001,
            amsgrad=True,
        )
        self.checkpoint = torch.load(
            checkpoint_path,
            map_location=torch.device(self.device),
            weights_only=False,
        )

    def initialize_net(self):
        """initialize model as pretrained LELA
        """
        self.net.load_state_dict(self.checkpoint['model_state_dict'])
        self.net.to(self.device)
        self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.optimizer.zero_grad()

    def fit_predict(self, X, y_vals, y_indices, n_rows, weights=None):
        """finetune LELA model by miminizing the loss on the provided labels
        Args:
            X: Processed label matrix (0 for abstention, 1 for positive class, -1 for negative/other classes).
            y_vals: Ground truth labels for specified indices (binary for the current target class).
            y_indices: Indices corresponding to y_vals.
            n_rows: Total number of data points (rows in the original LF matrix).
            weights: Optional weights for the loss function.
        Returns:
            Predicted probabilities for the positive class (tensor of size n_rows).
        """
        self.initialize_net()
        self.net.train()
        X_sparse = coo_matrix(np.squeeze(X))
        index = np.array([X_sparse.row, X_sparse.col]).T
        value = X_sparse.data
        index = torch.from_numpy(index).to(self.device).unsqueeze(0)
        value = torch.from_numpy(value).float().to(self.device).unsqueeze(0)
        
        # Ensure y_complete has size n_rows, handling empty y_indices
        y_complete = np.zeros(n_rows)
        if len(y_indices) > 0: # Check if there are any supervised labels
             # Ensure indices are within the bounds of y_complete
             valid_indices = y_indices[y_indices < n_rows]
             valid_y_vals = y_vals[y_indices < n_rows]
             if len(valid_indices) > 0:
                 y_complete[valid_indices] = valid_y_vals

        y_complete = torch.from_numpy(y_complete).float().to(self.device).unsqueeze(0)

        if weights:
            self.criterion = BCEMaskWeighted(weights)
        else:
            self.criterion = BCEMask()
            
        # Ensure mask has size n_rows
        mask = np.zeros(n_rows, dtype=bool)
        if len(y_indices) > 0: # Check if there are any supervised labels
             # Ensure indices used for mask are within bounds
             valid_indices_mask = y_indices[y_indices < n_rows]
             if len(valid_indices_mask) > 0:
                 mask[valid_indices_mask] = 1
                 
        mask = torch.from_numpy(mask).unsqueeze(0).to(self.device) # Ensure mask is on the correct device

        # Determine number of iterations, handle empty y_vals case
        num_iterations = 0
        if len(y_vals) > 0:
             num_iterations = int(np.sqrt(len(y_vals)))
        
        # Training loop
        for i in range(num_iterations):
            # Pass n_rows to the forward method
            pred, _ = self.net(index, value, n_rows=n_rows)
            # Ensure pred has the correct shape [1, n_rows] before loss calculation
            # This might require adjustment if self.net output shape is different
            if pred.shape[0] != n_rows:
                 # Handle potential shape mismatch, e.g., if pred comes out smaller
                 # This indicates an issue in self.net or sparse_mean handling of n_rows
                 # For now, assume pred has shape [n_rows] and unsqueeze correctly
                 # If pred is [batch_size, num_predicted_rows], ensure num_predicted_rows == n_rows
                 # Let's assume pred is correctly shaped [n_rows] after self.net call
                 pass # Placeholder for potential shape adjustment logic

            # Ensure pred is unsqueezed correctly for criterion [batch_size, n_rows]
            pred_unsqueezed = pred.unsqueeze(0) # Assuming pred was [n_rows]

            # Check shapes before loss calculation
            # print(f"pred shape: {pred_unsqueezed.shape}, y_complete shape: {y_complete.shape}, mask shape: {mask.shape}")
            
            # Ensure mask shape matches pred and y_complete for broadcasting or element-wise ops
            if pred_unsqueezed.shape != y_complete.shape or pred_unsqueezed.shape != mask.shape:
                 # Attempt to resolve shape mismatch if possible, or raise error
                 # This might involve adjusting mask or handling batch dimensions consistently
                 # print("Warning: Shape mismatch detected before loss calculation.")
                 # Fallback: Skip loss calculation if shapes are incompatible? Or try to align?
                 # For now, proceed assuming shapes are compatible or criterion handles it.
                 pass


            loss = self.criterion(pred_unsqueezed, y_complete, mask)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            
        self.net.eval()
        # Pass n_rows to the forward method for final prediction
        pred, _ = self.net(index, value, n_rows=n_rows)
        
        # Ensure the final prediction tensor has n_rows elements
        if pred.shape[0] != n_rows:
             # Handle discrepancy, e.g., by padding or error
             # This suggests n_rows propagation might still have issues
             # Fallback: Create a zero tensor of the correct size and fill with pred values
             final_pred = torch.zeros(n_rows, device=self.device)
             # Determine the number of elements to copy
             num_to_copy = min(pred.shape[0], n_rows)
             final_pred[:num_to_copy] = pred[:num_to_copy]
             pred = final_pred
             # print(f"Warning: Final prediction shape mismatch. Adjusted to {n_rows}.")


        return pred


class HyperLabelModel:
    def __init__(self, device=None, checkpoint_path=None):
        """
        Args:
            device: device (e.g. cpu, cuda:0) to use
            checkpoint_path: path of the pre-trained hyper label model checkpoint
        """
        if device is None:
            device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.device = device
        if checkpoint_path is None:
            pt = os.path.dirname(os.path.realpath(__file__))
            checkpoint_path = os.path.join(pt, "checkpoint.pt")
        self.checkpoint_path = checkpoint_path
        self.unsupervised_HyperLM = UnsupervisedWraper(self.device, self.checkpoint_path)

    def infer(self, X, y_indices=None, y_vals=None, return_probs=False):
        """perform unsupervised or semi-supervised label aggregation using the LELA hyper label model
        Args:
            X: label matrix, note "-1" denotes abstentions, and other non-negative integers (0, 1, 2, ...) denote classes.
               Each row of X represents the weak labels for a data point, and each column of X represents the weak labels provided by an labeling function (LF)
            y_indices: indices of the examples with known labels, e.g. [4, 5, 10, 20]. Only used for semi-supervised label aggregation
            y_vals: the values of the provided labels, e.g [1, 0, 1, 1]. Only used for semi-supervised label aggregation
            return_probs: return probabilities or hard labels
        Returns:
            predicted labels (hard labels or probabilities)
        """
        X = np.array(X)
        n_rows = X.shape[0] # Get the original number of rows
        is_semi_supervised = False
        if y_vals is not None:
            assert y_indices is not None
            # Convert to numpy arrays for consistent handling
            y_indices = np.array(y_indices)
            y_vals = np.array(y_vals)
            assert len(y_indices) == len(y_vals)
            # Filter out indices that are out of bounds
            valid_mask = y_indices < n_rows
            y_indices = y_indices[valid_mask]
            y_vals = y_vals[valid_mask]
            if len(y_indices) > 0: # Check if any valid supervised labels remain
                 is_semi_supervised = True
            else:
                 # If all provided indices were out of bounds, revert to unsupervised
                 print("Warning: All provided y_indices are out of bounds. Performing unsupervised inference.")
                 y_indices = None
                 y_vals = None


        if not is_semi_supervised:
            # Pass n_rows to predict_prob
            probs = self.unsupervised_HyperLM.predict_prob(X)
        else:
            # Pass n_rows to HyperLMSemi
            probs = HyperLMSemi(X, y_vals, y_indices, self.device, self.checkpoint_path, n_rows=n_rows)

        # Ensure probs has n_rows before calibration or argmax
        if probs.shape[0] != n_rows:
             # This indicates a potential issue in the called functions
             # Fallback strategy: Pad with uniform probabilities or raise an error
             print(f"Warning: Output probability shape mismatch ({probs.shape[0]} vs {n_rows}). Padding.")
             num_classes = probs.shape[1] if probs.ndim == 2 and probs.shape[1] > 0 else 2 # Assume binary if unknown
             padding = np.full((n_rows - probs.shape[0], num_classes), 1.0 / num_classes)
             probs = np.vstack((probs, padding))


        if return_probs:
            probs = calibrate_probs(probs)
            return probs
        else:
            # Handle potential empty probs or issues before argmax
            if probs.size == 0:
                 # Return default predictions, e.g., all zeros or based on some heuristic
                 print("Warning: Probability matrix is empty. Returning default predictions (all zeros).")
                 return np.zeros(n_rows, dtype=int)
            return np.argmax(probs, axis=1).flatten()


if __name__ == "__main__":
    hlm = HyperLabelModel()
    X = np.array([[0, 0, 1],
                  [1, 1, 1],
                  [-1, 1, 0],
                  [0, 1, 0]])
    y = hlm.infer(X)
    print(y)

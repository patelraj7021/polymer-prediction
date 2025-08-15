import torch
import numpy as np
import pandas as pd
from rdkit import Chem


def character_tokenizer(data_column):
    max_seq_length_index = data_column.str.len().idxmax()
    max_seq_length = len(data_column.iloc[max_seq_length_index]) + 2 # +2 for bos and eos
    char_index_map = charmap_from_text(data_column)
    result_tensor = tokens_from_charmap(data_column, char_index_map, max_seq_length)
    return result_tensor, char_index_map


def charmap_from_text(data_column):
    chars = set()
    # get all unique characters
    for row in data_column:
        chars = chars.union(set(row))
    i = 0
    char_index_map = {}
    # map characters to indices
    for char in chars:
        char_index_map[char] = i
        i += 1
    # add indices for bos, eos, unk
    char_index_map['<bos>'] = i 
    i += 1
    char_index_map['<eos>'] = i
    i += 1
    char_index_map['<unk>'] = i
    return char_index_map


def tokens_from_charmap(data_column, char_index_map, max_seq_length):
    result = []
    for row in data_column:
        # a record here is a SMILES sequence
        new_record = []
        # add bos
        new_tensor = torch.zeros(len(char_index_map))
        new_tensor[char_index_map['<bos>']] = 1
        new_record.append(new_tensor)
        for char in row:
            new_tensor = torch.zeros(len(char_index_map))
            if char in char_index_map:
                new_tensor[char_index_map[char]] = 1
            else:
                # accounts for unknown characters
                new_tensor[char_index_map['<unk>']] = 1
            new_record.append(new_tensor)
        # add eos
        new_tensor = torch.zeros(len(char_index_map))
        new_tensor[char_index_map['<eos>']] = 1
        new_record.append(new_tensor)
        record_tensor = torch.stack(new_record, dim=0)
        seq_len_diff = max_seq_length - len(record_tensor)
        if seq_len_diff > 0:
            pad_tensor = torch.zeros(seq_len_diff, len(char_index_map))
            record_tensor = torch.cat([record_tensor, pad_tensor], dim=0)
        result.append(record_tensor)
    result_tensor = torch.stack(result, dim=0)
    return result_tensor


# normalize the input tensor by the maximum value of each column
def normalize(x, eps=1e-8):
    max_abs_vals = torch.max(torch.nan_to_num(torch.abs(x)), dim=0).values + eps
    return torch.div(x, max_abs_vals), max_abs_vals


# tensor version of score function from kaggle
def wMAE(pred, target, n_i):
    # normalize across range of values for each property
    # values from train data
    # hardcoded to match with score function on kaggle
    r_i = [6.2028e+02, 5.5010e-01, 4.7750e-01, 1.0923e+00, 2.4945e+01]
    r_i = torch.tensor(r_i).to(pred.device)
    # count number of non-nan values for each property
    # n_i = torch.sum(~torch.isnan(target), dim=0)
    # if there are no non-nan values for a property, set the weight to 0
    # this happens often when batch size is small
    inverse_sqrt_n_i = torch.nan_to_num(1 / torch.sqrt(n_i), nan=0, posinf=0, neginf=0)
    property_weights = torch.div(inverse_sqrt_n_i, torch.sum(inverse_sqrt_n_i)) * inverse_sqrt_n_i.shape[0]
    diff = torch.abs(pred - target)
    scaled_diff = torch.div(diff, r_i)
    mean_scaled_diff = torch.nanmean(scaled_diff, dim=0).to(torch.float32)
    return torch.nanmean(torch.mul(torch.nan_to_num(mean_scaled_diff), property_weights), dim=0)


# pandas version of score function from kaggle
def wMAE_kaggle(solution, submission, row_id_column_name):
        # These values are from the train data.
    MINMAX_DICT =  {
            'Tg': [-148.0297376, 472.25],
            'FFV': [0.2269924, 0.77709707],
            'Tc': [0.0465, 0.524],
            'Density': [0.748691234, 1.840998909],
            'Rg': [9.7283551, 34.672905605],
        }
    NULL_FOR_SUBMISSION = -9999


    def scaling_error(labels, preds, property):
        error = np.abs(labels - preds)
        min_val, max_val = MINMAX_DICT[property]
        label_range = max_val - min_val
        return np.mean(error / label_range)


    def get_property_weights(labels):
        property_weight = []
        for property in MINMAX_DICT.keys():
            valid_num = np.sum(~np.isnan(labels[property]))
            property_weight.append(valid_num)
        property_weight = np.array(property_weight)
        property_weight = np.sqrt(1 / property_weight)
        return (property_weight / np.sum(property_weight)) * len(property_weight)


    def score(solution: pd.DataFrame, submission: pd.DataFrame, row_id_column_name: str) -> float:
        """
        Compute weighted Mean Absolute Error (wMAE) for the Open Polymer challenge.

        Expected input:
        - solution and submission as pandas.DataFrame
        - Column 'id': unique identifier for each sequence
        - Columns 'Tg', 'FFV', 'Tc', 'Density', 'Rg' as the predicted targets

        Examples
        --------
        >>> import pandas as pd
        >>> row_id_column_name = "id"
        >>> solution = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4})
        >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.5]*4, 'FFV': [0.5]*4, 'Tc': [0.5]*4, 'Density': [0.5]*4, 'Rg': [0.5]*4})
        >>> round(score(solution, submission, row_id_column_name=row_id_column_name), 4)
        0.2922
        >>> submission = pd.DataFrame({'id': range(4), 'Tg': [0.2]*4, 'FFV': [0.2]*4, 'Tc': [0.2]*4, 'Density': [0.2]*4, 'Rg': [0.2]*4} )
        >>> score(solution, submission, row_id_column_name=row_id_column_name)
        0.0
        """
        chemical_properties = list(MINMAX_DICT.keys())
        property_maes = []
        property_weights = get_property_weights(solution[chemical_properties])
        for property in chemical_properties:
            is_labeled = solution[property] != NULL_FOR_SUBMISSION
            property_maes.append(scaling_error(solution.loc[is_labeled, property], submission.loc[is_labeled, property], property))
        if len(property_maes) == 0:
            raise RuntimeError('No labels')
        return float(np.average(property_maes, weights=property_weights)) 
    return score(solution, submission, row_id_column_name)


def MSE(pred, target):
    diff = torch.pow(torch.nan_to_num(pred) - torch.nan_to_num(target), 2)
    loss = torch.nanmean(diff)
    return loss


def MAE(pred, target):
    diff = torch.abs(torch.nan_to_num(pred) - torch.nan_to_num(target))
    loss = torch.nanmean(diff)
    return loss


def make_smile_canonical(smile): # To avoid duplicates, for example: canonical '*C=C(*)C' == '*C(=C*)C'
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smile, str) or len(smile) == 0:
        return None
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]',
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])',
    ]
    # Check for any bad patterns
    for pattern in bad_patterns:
        if pattern in smile:
            return np.nan
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smile and any(x in smile for x in ['[R', 'R]']):
        return np.nan
    try:
        mol = Chem.MolFromSmiles(smile)
        canon_smile = Chem.MolToSmiles(mol, canonical=True)
        return canon_smile
    except:
        return np.nan

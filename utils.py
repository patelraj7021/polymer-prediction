import torch

def character_tokenizer(data_column):
    max_seq_length_index = data_column.str.len().idxmax()
    max_seq_length = len(data_column.iloc[max_seq_length_index])
    chars = set()
    for row in data_column:
        chars = chars.union(set(row))
    i = 0
    char_index_map = {}
    for char in chars:
        char_index_map[char] = i
        i += 1
    result = []
    for row in data_column:
        new_record = []
        for char in row:
            new_tensor = torch.zeros(len(chars))
            new_tensor[char_index_map[char]] = 1
            new_record.append(new_tensor)
        record_tensor = torch.stack(new_record, dim=0)
        seq_len_diff = max_seq_length - len(record_tensor)
        if seq_len_diff > 0:
            pad_tensor = torch.zeros(seq_len_diff, len(chars))
            record_tensor = torch.cat([record_tensor, pad_tensor], dim=0)
        result.append(record_tensor)
    result_tensor = torch.stack(result, dim=0)
    return result_tensor
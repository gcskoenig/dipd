def remove_string_from_list(data: list, target_strings: list[str], level: int = 0) -> list:
    if isinstance(data, list):
        res = []
        for item in data:
            if isinstance(item, list):
                res.append(remove_string_from_list(item, target_strings, level + 1))
            else:
                if item in target_strings:
                    if level == 0:
                        res.append([])
                else:
                    res.append(item)
        return res
    else:
        raise ValueError('data must be a list')
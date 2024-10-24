#!/usr/bin/env python
# coding: utf-8

# In[2]:


from typing import List

def reverse_by_n_elements(lst: List[int], n: int) -> List[int]:
    """
    Reverses the input list by groups of n elements.
    """
    result = []
    i = 0

    while i < len(lst):
        group = []
        
        for j in range(i, min(i + n, len(lst))):
            group.append(lst[j])
        
        
        for k in range(len(group) - 1, -1, -1):
            result.append(group[k])
        
        i += n

    return result

# Example usage
print(reverse_by_n_elements([1, 2, 3, 4, 5, 6, 7, 8], 3))  # Output: [3, 2, 1, 6, 5, 4, 8, 7]
print(reverse_by_n_elements([1, 2, 3, 4, 5], 2))  # Output: [2, 1, 4, 3, 5]
print(reverse_by_n_elements([10, 20, 30, 40, 50, 60, 70], 4))  # Output: [40, 30, 20, 10, 70, 60, 50]


# In[7]:


from typing import List, Dict

def group_by_length(lst: List[str]) -> Dict[int, List[str]]:
    """
    Groups the strings by their length and returns a dictionary.
    """
    length_dict = {}

    for string in lst:
        length = len(string)
        
        if length not in length_dict:
            length_dict[length] = []
       
        length_dict[length].append(string)

    
    return dict(sorted(length_dict.items()))

# Example usage
print(group_by_length(["apple", "bat", "car", "elephant", "dog", "bear"]))
# Output: {3: ['bat', 'car', 'dog'], 4: ['bear'], 5: ['apple'], 8: ['elephant']}

print(group_by_length(["one", "two", "three", "four"]))
# Output: {3: ['one', 'two'], 4: ['four'], 5: ['three']}


# In[8]:


from typing import Dict, Any

def flatten_dict(nested_dict: Dict[Any, Any], sep: str = '.') -> Dict[str, Any]:
   
    def _flatten(d: Dict[Any, Any], parent_key: str = '') -> Dict[str, Any]:
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                
                items.extend(_flatten(v, new_key).items())
            elif isinstance(v, list):
                
                for i, item in enumerate(v):
                    items.extend(_flatten({f"{new_key}[{i}]": item}).items())
            else:
                
                items.append((new_key, v))
        return dict(items)
    
   
    return _flatten(nested_dict)

# Example usage
nested_dict = {
    "road": {
        "name": "Highway 1",
        "length": 350,
        "sections": [
            {
                "id": 1,
                "condition": {
                    "pavement": "good",
                    "traffic": "moderate"
                }
            }
        ]
    }
}

flattened_dict = flatten_dict(nested_dict)
for key, value in flattened_dict.items():
    print(f"{key}: {value}")


# In[10]:


from typing import List

def unique_permutations(nums: List[int]) -> List[List[int]]:
    
    def backtrack(path: List[int], used: List[bool]):
        if len(path) == len(nums):
            result.append(path[:])
            return
        
        for i in range(len(nums)):
            
            if used[i] or (i > 0 and nums[i] == nums[i - 1] and not used[i - 1]):
                continue
            
            
            used[i] = True
            path.append(nums[i])
            
            
            backtrack(path, used)
            
            
            used[i] = False
            path.pop()

    
    nums.sort()
    result = []
    used = [False] * len(nums)
    backtrack([], used)
    
    return result


nums = [1, 1, 2]
output = unique_permutations(nums)
for perm in output:
    print(perm)


# In[12]:


import re
from typing import List

def find_all_dates(text: str) -> List[str]:
   
    
    patterns = [
        r'\b\d{2}-\d{2}-\d{4}\b',  
        r'\b\d{2}/\d{2}/\d{4}\b',  
        r'\b\d{4}\.\d{2}\.\d{2}\b'  
    ]
    
    
    combined_pattern = '|'.join(patterns)
    
    
    valid_dates = re.findall(combined_pattern, text)
    
    return valid_dates


text = "I was born on 23-08-1994, my friend on 08/23/1994, and another one on 1994.08.23."
output = find_all_dates(text)
print(output)


# In[15]:


get_ipython().system('pip install polyline')



# In[18]:


import polyline
import pandas as pd
import numpy as np

def haversine(lat1, lon1, lat2, lon2):
   
    
    R = 6371000
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def polyline_to_dataframe(polyline_str: str) -> pd.DataFrame:
  
    
    
    decoded = polyline.decode(polyline_str)
    
    
    df = pd.DataFrame(decoded, columns=['latitude', 'longitude'])
    
   
    distances = [0]  
    for i in range(1, len(df)):
        dist = haversine(df['latitude'][i - 1], df['longitude'][i - 1], 
                         df['latitude'][i], df['longitude'][i])
        distances.append(dist)
    
    
    df['distance'] = distances
    
    return df


polyline_str = "u{~vFzjqp@|CzOq@lE"
df = polyline_to_dataframe(polyline_str)
print(df)


# In[19]:


from typing import List

def rotate_matrix(matrix: List[List[int]]) -> List[List[int]]:
   
    n = len(matrix)
   
    rotated = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            rotated[j][n - 1 - i] = matrix[i][j]
    
    return rotated

def rotate_and_multiply_matrix(matrix: List[List[int]]) -> List[List[int]]:
   
    
    rotated_matrix = rotate_matrix(matrix)
    
   
    n = len(rotated_matrix)
    final_matrix = [[0] * n for _ in range(n)]
    
    for i in range(n):
        for j in range(n):
            
            row_sum = sum(rotated_matrix[i]) - rotated_matrix[i][j]
            col_sum = sum(rotated_matrix[k][j] for k in range(n)) - rotated_matrix[i][j]
            final_matrix[i][j] = row_sum + col_sum
            
    return final_matrix


matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
result = rotate_and_multiply_matrix(matrix)
for row in result:
    print(row)


# In[20]:


import pandas as pd


# In[21]:


df=pd.read_csv('dataset-1.csv')


# In[22]:


df


# In[ ]:





# In[28]:


import pandas as pd

def check_time_data_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Checks the completeness of timestamps for each (id, id_2) pair in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the columns id, id_2, startDay, startTime, endDay, endTime.
    
    Returns:
        pd.Series: A boolean series indicating if each (id, id_2) pair has incorrect timestamps.
    """
    # Create a combined datetime column for easier processing, with error handling
    df['start_timestamp'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], errors='coerce')
    df['end_timestamp'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], errors='coerce')

    # Check for any rows where timestamps couldn't be parsed
    if df['start_timestamp'].isnull().any() or df['end_timestamp'].isnull().any():
        print("Warning: Some dates could not be parsed. Check the input data.")
        df = df.dropna(subset=['start_timestamp', 'end_timestamp'])

    # Group by (id, id_2)
    grouped = df.groupby(['id', 'id_2'])

    # Function to check completeness for each group
    def check_group(group: pd.DataFrame) -> bool:
        # Get the unique days covered by the timestamps
        unique_days = group['start_timestamp'].dt.day_name().unique()
        
        # Check if the group is empty after dropping invalid timestamps
        if group.empty:
            return True  # Mark as incorrect if the group is empty

        # Check for the time range in the timestamps
        time_covered = pd.date_range(start=group['start_timestamp'].min().normalize(),
                                      end=group['end_timestamp'].max().normalize() + pd.Timedelta(days=1),
                                      freq='D')
        
        # Check if all 7 days are covered
        days_complete = set(unique_days) == {'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'}
        
        # Check if time covers a full 24-hour period
        full_day_covered = (group['start_timestamp'].min().time() == pd.Timestamp('00:00:00').time() and 
                            group['end_timestamp'].max().time() == pd.Timestamp('23:59:59').time())
        
        return not (days_complete and full_day_covered)

    # Apply the check to each group and create a boolean series
    result = grouped.apply(check_group)

    # Convert the result to a DataFrame with MultiIndex
    result_df = result.reset_index(drop=False)  # drop=False to keep index columns
    
    # Rename the resulting column (the result of apply will be named 0)
    result_df.rename(columns={0: 'incorrect_timestamps'}, inplace=True)

    # Set multi-index (id, id_2)
    result_df.set_index(['id', 'id_2'], inplace=True)
    
    return result_df['incorrect_timestamps']

# Example usage
# Load the dataset (ensure you have the correct path to dataset-1.csv)
df = pd.read_csv('dataset-1.csv')

# Check the timestamps
incorrect_timestamps = check_time_data_completeness(df)
print(incorrect_timestamps)


# In[ ]:





# In[ ]:





# In[ ]:





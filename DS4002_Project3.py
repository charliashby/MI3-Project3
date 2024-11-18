#!/usr/bin/env python
# coding: utf-8

# In[1]:


import zipfile

zip_file_path = "Images.zip"
output_dir = "Images"

with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(output_dir)


# Import Packages

# In[3]:


from torchvision import datasets
from collections import Counter
import matplotlib.pyplot as plt
import torch
from PIL import Image
import pandas as pd
import numpy as np
import os
import re


# EDA

# In[16]:


# Define the path to your dataset
data_dir = 'Images/Images'

# Load dataset using ImageFolder
dataset = datasets.ImageFolder(root=data_dir)

# Extract filenames from dataset.samples (which contains tuples of (image_path, label))
filenames = [os.path.basename(sample[0]) for sample in dataset.samples]


# In[21]:


# Function to strip numbers, remove .jpg, and replace underscores with spaces
def extract_breed(filename):
    # Remove numbers
    breed = re.sub(r'\d+', '', filename)
    # Remove .jpg extension
    breed = breed.replace('.jpg', '')
    # Replace underscores with spaces
    breed = breed.replace('_', ' ')
    return breed

# Apply the function to all filenames
breeds = [extract_breed(fname) for fname in filenames]


# In[28]:


# Create a DataFrame with 'filename' and 'breed' columns
df = pd.DataFrame({
    'filename': filenames,
    'breed': breeds
})

print(df.head())  # Display first few rows of the DataFrame


# In[29]:


# Assuming df is your DataFrame with a 'breed' column
unique_breeds = df['breed'].unique()

# Print all the unique breeds
print("List of unique breeds:")
for breed in unique_breeds:
    print(breed)
    
num_breeds = df['breed'].nunique()
print(num_breeds)


# In[30]:


# Count the number of images for each breed
breed_counts = df['breed'].value_counts()

# Plot a bar chart of the number of images per breed
plt.figure(figsize=(10, 6))  # Set figure size for better readability
breed_counts.plot(kind='bar')
plt.title('Number of Images per Breed')
plt.xlabel('Breed')
plt.ylabel('Number of Images')
plt.xticks(rotation=90)  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Show plot
plt.show()


# In[31]:


# Get the 5 breeds with the lowest number of images
lowest_5_breeds = breed_counts.nsmallest(5)

# Get the 5 breeds with the highest number of images
highest_5_breeds = breed_counts.nlargest(5)

# Display results
print("Breeds with the lowest number of images:")
print(lowest_5_breeds)

print("\nBreeds with the highest number of images:")
print(highest_5_breeds)


# In[32]:


combined_breeds = pd.concat([lowest_5_breeds, highest_5_breeds])

# Plot a bar chart
plt.figure(figsize=(10, 6))  # Set figure size for better readability
combined_breeds.plot(kind='bar', color='skyblue')
plt.title('Top 5 and Bottom 5 Breeds by Number of Images')
plt.xlabel('Breed')
plt.ylabel('Number of Images')
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for better readability
plt.tight_layout()  # Adjust layout to prevent clipping of labels

# Show plot
plt.show()


# Analysis

# In[33]:


# Create a bar chart to show the total number of breeds
plt.figure(figsize=(6, 4))
plt.bar(['Total Breeds'], [num_breeds], color='skyblue')
plt.title('Total Number of Unique Breeds')
plt.ylabel('Number of Breeds')

# Show plot
plt.show()


# Analysis

# In[6]:


import os
import pandas as pd
from glob import glob

# Define the path to your dataset
dataset_dir = 'Images/Images'

# Initialize lists to store file paths and labels (breeds)
file_paths = []
labels = []

# Loop through each breed folder
for breed in os.listdir(dataset_dir):
    breed_path = os.path.join(dataset_dir, breed)
    
    # Check if it's a directory (i.e., a breed folder)
    if os.path.isdir(breed_path):
        # Get all image file paths in the current breed folder
        images = glob(os.path.join(breed_path, "*.jpg"))  # Adjust extension if needed
        
        # Append file paths and labels to lists
        for img_path in images:
            file_paths.append(img_path)
            labels.append(breed)

# Create a pandas DataFrame with two columns: 'file_path' and 'breed'
df = pd.DataFrame({
    'file_path': file_paths,
    'breed': labels
})

# Display the first few rows of the dataframe
print(df.head())

# Optionally, save the dataframe to a CSV file for future use
df.to_csv('image_dataset.csv', index=False)


# In[9]:


# Function to clean the breed names
def clean_breed_name(breed):
    # Remove numbers using regex
    breed = re.sub(r'\d+', '', breed)
    # Remove the .jpg extension
    breed = breed.replace('.jpg', '')
    # Replace underscores with spaces
    breed = breed.replace('_', ' ')
    # Remove "n-" prefix if it exists
    breed = breed.replace('n-', '')
    return breed.strip()  # Remove any leading/trailing whitespace

# Apply the function to the 'breed' column
df['breed'] = df['breed'].apply(clean_breed_name)

# Display the cleaned dataframe
print(df.head())
df.to_csv('image_dataset.csv', index=False)


# In[14]:


# Add a new column 'image_id' with sequential IDs like "image_1", "image_2", etc.
df['image_id'] = ['image_' + str(i + 1) for i in range(len(df))]

# Display the updated dataframe with 'image_id' column
print(df.head())

# Optionally, save the updated dataframe to a CSV file for future use
df.to_csv('image_dataset.csv', index=False)


# In[22]:


get_ipython().system(' git clone https://github.com/charliashby/MI3-Project3.git')


# In[17]:





# In[18]:





# In[19]:





# In[20]:





# In[ ]:





# In[ ]:





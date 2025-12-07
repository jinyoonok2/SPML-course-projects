#!/bin/bash

# Download raw_data.zip from Google Drive
echo "Downloading raw_data.zip from Google Drive..."
wget --no-check-certificate 'https://drive.google.com/uc?export=download&id=12GLTqRjI7uj4ygqr774yyLDajPEvdbRm' -O raw_data.zip

# Check if download was successful
if [ $? -ne 0 ]; then
    echo "Download failed. Trying with curl..."
    curl -L 'https://drive.google.com/uc?export=download&id=12GLTqRjI7uj4ygqr774yyLDajPEvdbRm' -o raw_data.zip
fi

# Extract the zip file
echo "Extracting raw_data.zip..."
unzip raw_data.zip

# Remove the zip file after extraction
echo "Cleaning up..."
rm raw_data.zip

echo "Done! raw_data directory is ready."

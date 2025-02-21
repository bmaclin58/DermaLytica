import h5py

# File path
file_path = r"E:\Capstone Skin Cancer Project\ISIC 2024\isic-2024-challenge\train-image.hdf5"

# Open the HDF5 file
with h5py.File(file_path, "r") as hdf:
    print("\n📊 Preview of Data in Each Dataset:\n")

    for dataset_name in hdf.keys():
        dataset = hdf[dataset_name]
        print(f"📄 Dataset: {dataset_name}, Shape: {dataset.shape}, Type: {dataset.dtype}")

        # Try to preview the first few elements
        try:
            preview = dataset[:5] if dataset.shape != () else dataset[()]  # Handle scalars
            print(f"  🔍 Sample data: {preview}\n")
        except Exception as e:
            print(f"  ⚠️ Could not read data: {e}\n")

import os
import cv2
import csv
import pandas as pd

def extract_quasi_identifiers(filename):
    components = os.path.splitext(filename)[0].split("_")

    if(len(components) == 4):
        # Extract the age, gender, and race from the filename
        age = int(components[0])
        gender = int(components[1])
        race = int(components[2])

        return age, gender, race
    
    else:
        print(f"Ignoring file {filename} with {len(components)} parts")
        return None, None, None

# Path to the directory containing the image files
data_dir = "./UTKFace"

# List all the files in the data directory
files = os.listdir(data_dir)

# Create a CSV file to store the quasi-identifiers and image paths
with open("quasi_identifiers.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)

    # Write the header row to the CSV file
    writer.writerow(["Age", "Gender", "Race", "Image Path"])

    for filename in files:
        # Extract the quasi-identifiers from the file name
        age, gender, race = extract_quasi_identifiers(filename)

        if age is None:
            continue

        filepath = os.path.join(data_dir, filename)

        writer.writerow([age, gender, race, filepath])

print("Done writing CSV file!")

# Load the CSV file with the quasi-identifiers and image paths
df = pd.read_csv("quasi_identifiers.csv")

# Group the images based on the quasi-identifiers using l-diversity with l=10
groups = df.groupby(["Age", "Gender", "Race"]).apply(lambda x: x.sample(n=max(10, len(x)), replace=True))

df.drop_duplicates(subset=["Image Path"], inplace=True)

groups = groups.reset_index(drop=True)

# Save the resulting groups to a new CSV file
groups.to_csv("l_diversified_groups.csv", index=False)

print("Done writing l-diversified groups CSV file!")


df = pd.read_csv("l_diversified_groups.csv")

os.makedirs("anonymized_images", exist_ok=True)

for i, group in df.groupby(["Age", "Gender", "Race"]):
    # Load each image in the group and apply blurring and pixelation
    for j, row in group.iterrows():
        image_path = row["Image Path"]
        image = cv2.imread(image_path)

        # Apply blurring
        blurred_image = cv2.GaussianBlur(image, (21, 21), 0)

        # Apply pixelation
        pixelated_image = cv2.resize(blurred_image, (32, 32), interpolation=cv2.INTER_LINEAR)
        pixelated_image = cv2.resize(pixelated_image, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Save the anonymized image to the new directory with a new file name
        new_image_path = os.path.join("anonymized_images", f"{i[0]}_{i[1]}_{i[2]}_{j}.jpg")
        cv2.imwrite(new_image_path, pixelated_image)

print("Done anonymizing images!")
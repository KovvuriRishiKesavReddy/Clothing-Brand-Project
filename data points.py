import pandas as pd
import numpy as np
import random

def generate_person_data_enhanced(num_data_points, num_exceptional_cases):
    data = []

    # Define ranges for height and weight (approximate for adults)
    height_min, height_max = 150, 190  # cm
    weight_min, weight_max = 45, 120   # kg

    # Define clothing sizes
    clothing_sizes_ordered = ["XS", "S", "M", "L", "XL", "XXL"]
    size_map = {size: i for i, size in enumerate(clothing_sizes_ordered)}

    # More diverse body shapes (still generalized)
    body_shapes = [
        "Rectangle", "Pear", "Apple", "Hourglass", "Inverted Triangle",
        "Spoon", "Diamond", "Oval", "Athletic"
    ]

    # Expanded and more specific color palettes
    color_palettes = {
        "Warm Tones": ["Red", "Orange", "Yellow", "Coral", "Maroon", "Gold"],
        "Cool Tones": ["Blue", "Green", "Purple", "Teal", "Navy", "Lavender"],
        "Neutrals": ["Black", "White", "Grey", "Beige", "Brown", "Cream"],
        "Vibrant Hues": ["Fuchsia", "Lime Green", "Electric Blue", "Bright Orange", "Turquoise", "Magenta"],
        "Earthy Tones": ["Olive Green", "Terracotta", "Mustard Yellow", "Rust", "Khaki", "Forest Green"],
        "Pastel Tones": ["Baby Pink", "Light Blue", "Mint Green", "Lavender", "Pale Yellow", "Peach"]
    }

    # Generalized "Style Preferences" (not linked to personality via height/size)
    style_preferences = {
        "Bold & Expressive": ["Vibrant Hues", "Warm Tones"],
        "Calm & Classic": ["Neutrals", "Cool Tones", "Earthy Tones"],
        "Soft & Gentle": ["Pastel Tones", "Neutrals"],
        "Adventurous & Natural": ["Earthy Tones", "Warm Tones", "Vibrant Hues"]
    }
    general_style_prefs = list(style_preferences.keys())

    # --- Logic for Hip Circumference and Shoulder Width ---
    base_hip_min, base_hip_max = 85, 110 # cm
    base_shoulder_min, base_shoulder_max = 38, 48 # cm (across the back)

    for i in range(num_data_points):
        # Generate core data
        height = round(random.uniform(height_min, height_max), 2)
        weight = round(random.uniform(weight_min, weight_max), 2)
        bmi = round(weight / ((height / 100)**2), 2)

        bmi_category = ""
        if bmi < 18.5:
            bmi_category = "Underweight"
        elif 18.5 <= bmi < 24.9:
            bmi_category = "Normal"
        elif 25 <= bmi < 29.9:
            bmi_category = "Overweight"
        else:
            bmi_category = "Obese"

        # Introduce exceptional cases for BMI
        if i < num_exceptional_cases:
            bmi_category_choice = random.choice(["Underweight", "Overweight", "Obese"])
            if bmi_category_choice == "Underweight":
                bmi = round(random.uniform(15, 18), 2)
            elif bmi_category_choice == "Overweight":
                bmi = round(random.uniform(25, 29), 2)
            else: # Obese
                bmi = round(random.uniform(30, 40), 2)
            weight = round(bmi * ((height / 100)**2), 2)
            bmi_category = bmi_category_choice

        body_shape = random.choice(body_shapes)

        # --- Generate Hip Circumference and Shoulder Width with some correlation ---
        hip_circumference = random.uniform(base_hip_min, base_hip_max)
        shoulder_width = random.uniform(base_shoulder_min, base_shoulder_max)

        hip_circumference += (weight - 70) * 0.2
        shoulder_width += (height - 170) * 0.05

        if body_shape in ["Pear", "Hourglass", "Spoon", "Apple", "Oval"]:
            hip_circumference *= random.uniform(1.05, 1.15)
            if body_shape == "Pear":
                shoulder_width *= random.uniform(0.9, 0.98)
        elif body_shape in ["Inverted Triangle", "Athletic"]:
            shoulder_width *= random.uniform(1.05, 1.15)
            if body_shape == "Inverted Triangle":
                hip_circumference *= random.uniform(0.9, 0.98)
        
        hip_circumference = round(np.clip(hip_circumference, 75, 130), 2)
        shoulder_width = round(np.clip(shoulder_width, 35, 55), 2)

        # --- NEW LOGIC FOR CLOTHING SIZE BASED ON FEATURES ---
        # Initialize a base size index (e.g., M is index 2)
        size_idx = size_map["M"]

        # Adjust based on Height
        if height < 160:
            size_idx -= 1 # Smaller for shorter
        elif height > 180:
            size_idx += 1 # Larger for taller

        # Adjust based on Weight
        if weight < 55:
            size_idx -= 1
        elif weight > 80:
            size_idx += 1
        elif weight > 95: # More aggressive for heavier
            size_idx += 1

        # Adjust based on Shoulder Width
        if shoulder_width < 40:
            size_idx -= 1
        elif shoulder_width > 46:
            size_idx += 1

        # Adjust based on BMI Category (fine-tuning)
        if bmi_category == "Underweight":
            size_idx -= 1
        elif bmi_category == "Overweight":
            size_idx += 1
        elif bmi_category == "Obese":
            size_idx += 2

        # Adjust based on Body Shape (simplified for overall size)
        if body_shape == "Inverted Triangle": # Broader shoulders
            size_idx += 0.5 # Slight bump up, might round to next size
        elif body_shape == "Athletic":
            size_idx += 0.5
        elif body_shape == "Pear": # Smaller top, larger bottom
            size_idx -= 0.5 # Slight bump down for overall average size
        elif body_shape == "Apple": # Larger midsection
            size_idx += 0.5

        # Clamp size_idx to valid range [0, len(clothing_sizes_ordered)-1]
        size_idx = int(np.clip(round(size_idx), 0, len(clothing_sizes_ordered) - 1))
        clothing_size = clothing_sizes_ordered[size_idx]
        # --- END NEW LOGIC FOR CLOTHING SIZE ---

        # Assign a general style preference
        assigned_style_pref = random.choice(general_style_prefs)

        # Based on style preference, choose a general color palette
        possible_palettes_for_style = style_preferences[assigned_style_pref]
        chosen_palette_name = random.choice(possible_palettes_for_style)
        
        # Then, pick a specific color from that chosen palette
        recommended_color = random.choice(color_palettes[chosen_palette_name])

        data.append([
            height, weight, bmi, bmi_category, body_shape,
            hip_circumference, shoulder_width,
            clothing_size, assigned_style_pref, recommended_color
        ])

    columns = [
        'Height_cm', 'Weight_kg', 'BMI', 'BMI_Category', 'Body_Shape',
        'Hip_Circumference_cm', 'Shoulder_Width_cm',
        'Clothing_Size', 'Style_Preference', 'Recommended_Cloth_Color'
    ]

    df = pd.DataFrame(data, columns=columns)
    return df

# --- Example usage to generate the CSV ---
num_total_points = 20000
num_exceptional = 1000

df_generated = generate_person_data_enhanced(num_total_points, num_exceptional)

# Save to CSV
output_csv_file = "data_points.csv" # New filename to reflect logical size
df_generated.to_csv(output_csv_file, index=False)

print(f"CSV file '{output_csv_file}' generated successfully with {num_total_points} data points.")
# print("\nFirst 5 rows of the generated data:")
# print(df_generated.head())
# print("\nDistribution of Clothing Sizes:")
# print(df_generated['Clothing_Size'].value_counts().sort_index())
# print("\nData types of the generated columns:")
# print(df_generated.info())
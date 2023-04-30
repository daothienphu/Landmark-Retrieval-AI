import streamlit as st
from PIL import Image
import os
from pathlib import Path
from model_lib import run_model
import random

image_dir = Path("data/database")
database_cutoff = 20
image_height = 10

def main():
    st.title("Landmark Retrieval AI")
    intro = "Upload an image of a landmark here, we will retrieve some more images of the landmark from our database. You can use an image from the Database Preview section below if you don't have one at hand."
    st.write(f'<div style="text-align: justify">{intro}</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"])
    res = ""
    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        st.image(image, use_column_width=True)

        res = run_model(task='app', image=image)
        
        st.title("Results")
        process_time = res["time"]
        st.write(f"Results found after {process_time:.06f} seconds.")
        res = res["res"]
        col1, col2, col3 = st.columns(3)
        for i in range(len(res)):
            name, confidence = res[i].split(": ")
            image = Image.open(str(os.path.join(image_dir, Path(f"{name}.jpg"))))
            with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
                st.image(image, use_column_width=True, caption=f"{name}.jpg, confidence: {confidence}")


    st.title("Database Preview")
    # Load multiple images
    images = []
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            images.append(str(Path(file)))
            if len(images) >= min(database_cutoff * 10, 1200):
                break

    images = [f"{image_dir}/{image}" for image in images]
    images = random.sample(images, k=database_cutoff)
    images = list(map(Image.open, images))
    
    col1, col2, col3 = st.columns(3)
    for i, image in enumerate(images):
        with col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3:
            st.image(image, use_column_width=True)

    footer = """
    <style>
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background-color: #222831;
            color: white;
            text-align: center;
            font-family: Arial, sans-serif;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 42px;
            font-size: 14px;
        }

        .footer p {
            margin: 0; /* remove the default margin of the <p> tag */
        }

        .footer a {
            color: white;
            text-decoration: underline; /* add an underline to the link */
        }
    </style>

    <div class="footer">
        <p>Made by <a href="https://github.com/daothienphu">SpookyFish</a> and <a href="https://github.com/peternguyen39">Stargazer</a>.</p>
    </div>
    """

    st.markdown(footer, unsafe_allow_html=True)

    

if __name__ == '__main__':
    main()
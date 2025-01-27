import streamlit as st
import requests
from PIL import Image, ImageDraw
import io

# Streamlit UI setup
st.title("License Plate Detection and Recognition")
st.header("Upload an image to detect and recognize license plate characters")

# File uploader
uploaded_file = st.file_uploader("Choose an image of a license plate", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")

    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Convert the image to bytes
    image_bytes = io.BytesIO()
    image.save(image_bytes, format="JPEG")
    image_bytes = image_bytes.getvalue()

    # Call the FastAPI endpoint
    st.write("Processing the image...")
    try:
        response = requests.post(
            "http://localhost:8000/ocr",  # Replace with your FastAPI endpoint URL
            files={"image": ("license_plate.jpg", image_bytes, "image/jpeg")}
        )

        if response.status_code == 200:
            result = response.json()
            st.write("### Recognized License Plate Characters:")

            recognized_boxes = result.get("recognized_texts", [])
            confidence_scores = result.get("confidence_scores", [])

            # Draw bounding boxes on the image
            draw = ImageDraw.Draw(image)

            for box, (text, confidence) in zip(recognized_boxes, confidence_scores):
                # Ensure box coordinates are flattened and properly formatted
                flat_box = [(int(coord[0]), int(coord[1])) for coord in box]

                # Draw the bounding box
                draw.polygon(flat_box, outline="red", width=2)
                # Annotate with text and confidence
                x, y = flat_box[0]  # Top-left corner of the box
                draw.text((x, y - 10), f"{text} ({confidence:.2f})", fill="red")

            # Display the annotated image
            st.image(image, caption="Processed Image with Annotations", use_container_width=True)

            # Display recognized text below the image
            st.write("### Text Recognized in Image:")
            for text, confidence in confidence_scores:
                st.write(f"- **Text**: {text}, **Confidence**: {confidence:.2f}")
        else:
            st.error(f"Error from FastAPI: {response.text}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")

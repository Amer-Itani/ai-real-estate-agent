import streamlit as st
import requests

st.set_page_config(page_title="AI Real Estate Agent", layout="centered")

st.title("🏠 AI Real Estate Price Predictor")

st.write("Enter a property description:")

query = st.text_area("Example: A good house with 1800 sqft, 2 garage in CollgCr")

if st.button("Analyze"):

    if not query.strip():
        st.warning("Please enter a query")
    else:
        try:
            response = requests.post(
                "http://127.0.0.1:8000/analyze-query",
                json={"query": query}
            )

            data = response.json()

            st.subheader("📊 Extracted Features")
            st.json(data["extracted_features"])

            if data["missing_features"]:
                st.warning(f"Missing features: {data['missing_features']}")
            else:
                st.success("All features extracted")

            st.subheader("📈 Prediction")

            if data["predicted_price"]:
                st.success(f"Predicted Price: ${data['predicted_price']:,.2f}")
            else:
                st.error("Prediction not available")

            st.subheader("🧠 Interpretation")
            st.write(data["interpretation"])

            st.caption(f"Prompt version used: {data['prompt_version']}")

        except Exception as e:
            st.error(f"Error: {e}")
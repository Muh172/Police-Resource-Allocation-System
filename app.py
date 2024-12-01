import streamlit as st
import torch
from transformers import (
    DPRQuestionEncoder,
    DPRContextEncoder,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import faiss
import numpy as np
import pandas as pd
import re
import logging
from typing import Optional, Dict, Tuple
import pydeck as pdk  # For advanced map visualization
from sklearn.cluster import KMeans  # For clustering
import altair as alt  # For bar chart visualization

# Suppress specific warnings
import warnings
warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch.nn.functional"
)

# ---------------------------
# Logging Configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("app.log"), logging.StreamHandler()],
)

# ---------------------------
# Device Configuration
# ---------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logging.info(f"Using device: {device}")

# ---------------------------
# Question Templates
# ---------------------------
# Define regex patterns for each template
TEMPLATES = {
    "highest_crime": re.compile(
        r"what\s+is\s+the\s+highest\s+crime\??", re.IGNORECASE
    ),
    "total_crimes": re.compile(
        r"what\s+is\s+the\s+total\s+number\s+of\s+(?P<crime_type>[A-Za-z\s]+)"
        r"\s+in\s+(?P<year>\d{4})\?",
        re.IGNORECASE,
    ),
    # Add more templates as needed
}

# ---------------------------
# Helper Functions for Templates
# ---------------------------

def is_general_crime_question(query: str) -> bool:
    """
    Determines if the query matches any of the predefined general crime question templates.
    """
    for pattern in TEMPLATES.values():
        if pattern.match(query.strip()):
            logging.debug(f"Query matched pattern: {pattern.pattern}")
            return True
    return False

def extract_parameters(query: str) -> Optional[Tuple[str, Dict[str, str]]]:
    """
    Extracts the template name and parameters from the query based on matching templates.
    """
    for template_name, pattern in TEMPLATES.items():
        match = pattern.match(query.strip())
        if match:
            params = match.groupdict()
            logging.debug(
                f"Extracted parameters: {params} using template: {template_name}"
            )
            return template_name, params
    return None

def generate_template_response(
    template_name: str, params: Dict[str, str], vector_database: "VectorDatabase"
) -> str:
    """
    Generates a response based on the extracted parameters by querying the data.
    """
    if template_name == "highest_crime":
        highest_crime = vector_database.get_highest_crime()
        if highest_crime:
            response = f"The highest crime in Chicago is {highest_crime}."
            logging.info(f"Generated response: {response}")
            return response
        else:
            logging.info("No crime data available for Chicago.")
            return "No crime data available for Chicago."

    elif template_name == "total_crimes":
        crime_type = params.get("crime_type").strip().lower()
        year = params.get("year")
        total = vector_database.get_total_crimes(crime_type, year)
        if total is not None:
            response = f"The total number of {crime_type} in {year} is {total}."
            logging.info(f"Generated response: {response}")
            return response
        else:
            logging.info(f"No data available for {crime_type} in {year}.")
            return f"No data available for {crime_type} in {year}."

    else:
        logging.warning(f"Unhandled template: {template_name}")
        return "I'm sorry, I couldn't process your request."

# ---------------------------
# DPR Retriever Class
# ---------------------------
class DPRRetriever:
    def __init__(self, faiss_index_path: str):
        """
        Initializes the DPRRetriever with pre-trained DPR models and a FAISS index.
        """
        try:
            # Load pre-trained DPR models
            logging.info("Loading DPR Question Encoder...")
            self.question_encoder = DPRQuestionEncoder.from_pretrained(
                "facebook/dpr-question_encoder-single-nq-base"
            ).to(device)
            logging.info("DPR Question Encoder loaded successfully.")

            logging.info("Loading DPR Context Encoder...")
            self.context_encoder = DPRContextEncoder.from_pretrained(
                "facebook/dpr-ctx_encoder-single-nq-base"
            ).to(device)
            logging.info("DPR Context Encoder loaded successfully.")

            # Load FAISS index
            logging.info(f"Loading FAISS index from {faiss_index_path}...")
            self.index = faiss.read_index(faiss_index_path)
            logging.info("FAISS index loaded successfully.")
        except Exception as e:
            logging.error(f"Error initializing DPRRetriever: {e}", exc_info=True)
            st.error("Failed to initialize the DPR Retriever.")
            raise e

    def retrieve(self, query_embedding: np.ndarray, k: int = 5) -> np.ndarray:
        """
        Searches the FAISS index with the query embedding and retrieves document IDs.
        """
        try:
            query_embedding = query_embedding.reshape(1, -1).astype("float32")
            distances, indices = self.index.search(query_embedding, k)
            logging.debug(f"Retrieved document indices: {indices}")
            return indices
        except Exception as e:
            logging.error(f"Error during FAISS search: {e}", exc_info=True)
            st.error("Failed to retrieve documents from the index.")
            return np.array([])

# ---------------------------
# Vector Database Class
# ---------------------------
class VectorDatabase:
    def __init__(self, embedding_file: str, csv_file: str):
        """
        Initializes the VectorDatabase with embeddings and crime data.
        """
        try:
            logging.info(f"Loading embeddings from {embedding_file}...")
            self.embeddings = np.load(embedding_file)
            logging.info("Embeddings loaded successfully.")

            logging.info(f"Loading crime data from {csv_file}...")
            self.crime_data = pd.read_csv(csv_file)
            logging.info("Crime data loaded successfully.")

            # Rename coordinate columns if necessary
            self.crime_data.rename(
                columns={"lat": "lat", "lon": "lon"}, inplace=True
            )

            # Data Validation
            self.validate_data()

        except Exception as e:
            logging.error(f"Error initializing VectorDatabase: {e}", exc_info=True)
            st.error("Failed to initialize the Vector Database.")
            raise e

    def validate_data(self):
        """
        Validates the crime data for anomalies before processing.
        """
        # Ensure correct data types
        if 'crime_count' in self.crime_data.columns:
            self.crime_data["crime_count"] = pd.to_numeric(
                self.crime_data["crime_count"], errors="coerce"
            )
            self.crime_data["crime_count"].fillna(1, inplace=True)
            self.crime_data["crime_count"] = self.crime_data["crime_count"].astype(int)
            logging.info("Converted 'crime_count' to numeric type and filled NaNs with 1.")
        else:
            # If 'crime_count' doesn't exist, create it with default value 1
            self.crime_data["crime_count"] = 1
            logging.info("'crime_count' column created with default value 1.")

        self.crime_data["year"] = pd.to_numeric(
            self.crime_data["year"], errors="coerce"
        )
        self.crime_data["year"].fillna(0, inplace=True)
        self.crime_data["year"] = self.crime_data["year"].astype(int)
        logging.info("Converted 'year' to numeric type and filled NaNs with 0.")

        # Remove negative 'crime_count' entries
        initial_length = len(self.crime_data)
        self.crime_data = self.crime_data[self.crime_data["crime_count"] >= 0]
        removed_entries = initial_length - len(self.crime_data)
        if removed_entries > 0:
            logging.warning(
                f"Removed {removed_entries} entries with negative 'crime_count'."
            )

        # Remove duplicates
        duplicate_count = self.crime_data.duplicated().sum()
        if duplicate_count > 0:
            logging.warning(f"Found {duplicate_count} duplicate entries. Removing them.")
            self.crime_data.drop_duplicates(inplace=True)
        else:
            logging.info("No duplicate entries found.")

        # Check for unusually high 'crime_count' values
        max_crime_count = self.crime_data["crime_count"].max()
        if max_crime_count > 1000:  # Adjust threshold as appropriate
            logging.warning(
                f"Found unusually high 'crime_count' value: {max_crime_count}. Investigate data."
            )
        else:
            logging.info(
                f"All 'crime_count' values are within acceptable ranges (max={max_crime_count})."
            )

    def get_documents(self, document_ids: np.ndarray) -> pd.DataFrame:
        """
        Retrieves crime data rows corresponding to the given document IDs.
        """
        try:
            # Flatten the array and remove any negative indices
            flat_ids = document_ids.flatten()
            flat_ids = flat_ids[flat_ids >= 0]
            retrieved = self.crime_data.iloc[flat_ids].reset_index(drop=True)
            logging.debug(f"Retrieved {len(retrieved)} documents.")
            return retrieved
        except Exception as e:
            logging.error(f"Error retrieving documents: {e}", exc_info=True)
            st.error("Failed to retrieve documents from the Vector Database.")
            return pd.DataFrame()

    def get_highest_crime(self) -> Optional[str]:
        """
        Retrieves the crime type with the highest total count across all records.
        """
        try:
            if self.crime_data.empty:
                logging.info("Crime data is empty.")
                return None
            # Count occurrences of each crime_type
            crime_counts = self.crime_data['crime_type'].value_counts()
            # Identify the crime type with the maximum count
            highest_crime = crime_counts.idxmax()
            logging.debug(f"Highest crime in Chicago based on count: {highest_crime}")
            return highest_crime
        except Exception as e:
            logging.error(f"Error in get_highest_crime: {e}", exc_info=True)
            st.error("Failed to retrieve the highest crime data.")
            return None

    def get_total_crimes(self, crime_type: str, year: str) -> Optional[int]:
        """
        Retrieves the total number of specified crimes in a given year.
        """
        try:
            # Ensure 'crime_type' and 'year' are correctly formatted
            crime_type = crime_type.strip().lower()
            year = int(year)

            # Filter data
            filtered_data = self.crime_data[
                (self.crime_data["crime_type"].str.lower() == crime_type)
                & (self.crime_data["year"] == year)
            ]
            logging.debug(
                f"Number of records for {crime_type} in {year}: {filtered_data.shape[0]}"
            )

            if filtered_data.empty:
                logging.info(f"No data found for crime type: {crime_type} in year: {year}")
                return None

            # Calculate total crimes as number of records
            total = len(filtered_data)
            logging.debug(f"Total number of {crime_type} in {year}: {total}")

            # Additional Check: If total is unexpectedly high, log a warning
            if total > 100000:  # Adjust threshold as needed
                logging.warning(
                    f"Total number of {crime_type} in {year} is unusually high: {total}. Verify data integrity."
                )

            return total
        except Exception as e:
            logging.error(f"Error in get_total_crimes: {e}", exc_info=True)
            st.error("Failed to retrieve total crimes data.")
            return None

# ---------------------------
# LLM Class
# ---------------------------
class LLM:
    def __init__(self, model_name: str):
        """
        Initializes the LLM with the specified model.
        """
        try:
            logging.info(f"Loading tokenizer for {model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            logging.info(f"Tokenizer for {model_name} loaded successfully.")

            logging.info(f"Loading model {model_name}...")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
            logging.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logging.error(f"Error initializing LLM ({model_name}): {e}", exc_info=True)
            st.error(f"Failed to initialize the model: {model_name}.")
            raise e

    def generate_answer(self, input_text: str) -> str:
        """
        Generates an answer based on the input text using the loaded model.
        """
        try:
            input_ids = self.tokenizer.encode(
                input_text, return_tensors="pt"
            ).to(device)
            logging.debug("Input text tokenized successfully.")
            with torch.no_grad():
                output_ids = self.model.generate(input_ids, max_length=512)
            answer = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            logging.debug("Answer generated successfully.")
            return answer
        except Exception as e:
            logging.error(f"Error generating answer: {e}", exc_info=True)
            st.error("Failed to generate an answer.")
            return "I'm sorry, I couldn't generate a response at this time."

# ---------------------------
# Query Classification and Response Generation
# ---------------------------

def needs_rag(query: str) -> bool:
    """
    Determines whether to use the RAG pipeline based on the query.
    """
    bypass_keywords = ["deploy", "officers", "deployment"]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in bypass_keywords):
        logging.debug("Query contains bypass keywords.")
        return False  # Bypass RAG
    return not is_general_crime_question(query)

def get_response(
    query: str,
    retriever: DPRRetriever,
    vector_database: VectorDatabase,
    bypass_llm: LLM,
    rag_llm: LLM,
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> Optional[str]:
    """
    Generate a response based on the user's query using either templated responses, RAG, or bypass LLM.
    """
    try:
        if is_general_crime_question(query):
            st.info("Processing general crime statistics...")
            logging.info("General crime statistics query detected.")

            # Extract parameters
            extraction = extract_parameters(query)
            if not extraction:
                logging.warning("Failed to extract parameters from the query.")
                return "I'm sorry, I couldn't understand your query."

            template_name, params = extraction
            logging.debug(f"Extracted parameters: {params}")

            # Generate response based on template
            response = generate_template_response(
                template_name, params, vector_database
            )
            logging.info("Response generated.")

        elif not needs_rag(query):
            # Handle deployment-related queries with bypass_llm
            st.info("Generating a recommendation...")
            logging.info("Deployment-related query detected; generating a recommendation.")

            response = bypass_llm.generate_answer(query)
            logging.info("Recommendation generated.")

        else:
            # Handle general queries with RAG
            st.info("Using RAG pipeline...")
            logging.info("RAG pipeline initiated for query.")

            # Tokenize the query
            query_tokenized = tokenizer(query, return_tensors="pt").to(device)
            logging.debug("Query tokenization successful.")

            # Generate query embedding
            with torch.no_grad():
                query_embedding = retriever.question_encoder(**query_tokenized).pooler_output
            logging.debug("Query embedding generated.")

            # Retrieve relevant documents
            document_ids = retriever.retrieve(query_embedding.cpu().numpy(), k=k)
            if document_ids.size == 0:
                response = "No relevant information found for your query."
                logging.info("No documents retrieved; generated default response.")
                return response
            logging.info(f"Retrieved document IDs: {document_ids}")

            retrieved_rows = vector_database.get_documents(document_ids)
            logging.debug("Documents retrieved from the vector database.")

            if retrieved_rows.empty:
                logging.info("No documents retrieved from the vector database.")
                response_data = "No relevant information found for your query."
            else:
                # Clean 'crime_location' entries if necessary
                if "crime_location" in retrieved_rows.columns:
                    retrieved_rows = retrieved_rows.copy()
                    retrieved_rows["crime_location"] = retrieved_rows[
                        "crime_location"
                    ].apply(
                        lambda loc: "Unspecified Location"
                        if loc.lower() == "other location"
                        else loc
                    )
                    logging.debug("Crime locations normalized.")

                # Aggregate data by neighborhood
                summarized_data = (
                    retrieved_rows.groupby("neighborhood")
                    .agg({"crime_severity_index": "sum"})
                    .reset_index()
                    .sort_values(by="crime_severity_index", ascending=False)
                )
                logging.debug("Data aggregated by neighborhood.")

                # Generate response from summarized data
                if not summarized_data.empty:
                    response_data = "\n".join(
                        [
                            f"{int(row['crime_severity_index'])} total severity reported in {row['neighborhood']}."
                            for _, row in summarized_data.iterrows()
                        ]
                    )
                    logging.info("Response data prepared from summarized data.")
                else:
                    response_data = "No significant data available for your query."
                    logging.info("No significant data found for the query.")

            # Generate response using RAG-specific LLM
            response = rag_llm.generate_answer(response_data)
            logging.info("Response generated using RAG-specific LLM.")

        return response

    except Exception as e:
        logging.error(f"Error in get_response: {e}", exc_info=True)
        st.error("An error occurred while processing your query. Please try again later.")
        return None

# ---------------------------
# Streamlit App Initialization
# ---------------------------

def main():
    """
    Initializes and runs the Streamlit app for Police Resource Allocation System.
    """
    st.set_page_config(
        page_title="Police Resource Allocation System", layout="wide"
    )
    st.title("🚓 Police Resource Allocation System")

    st.sidebar.header("Configuration")

    # Sidebar inputs for file paths
    faiss_index_path = st.sidebar.text_input(
        "FAISS Index Path", "crime_faiss_index.index"
    )
    embedding_file = st.sidebar.text_input(
        "Embeddings File Path", "crime_embeddings.npy"
    )
    csv_file = st.sidebar.text_input(
        "CSV Data File Path", "Chicago_crime.csv"
    )
    bypass_llm_model = st.sidebar.text_input(
        "Bypass LLM Model Name",
        "Fine_Tuned_T5_Base",  # Example model name; adjust as needed
    )
    rag_llm_model = st.sidebar.text_input("RAG LLM Model Name", "t5-base")

    # Sidebar input for number of retrieved documents
    num_retrieved_docs = st.sidebar.number_input(
        "Number of Retrieved Documents (k)",
        min_value=1,
        max_value=100,
        value=5,
        step=1,
    )

    # Load models and databases with caching
    @st.cache_resource
    def load_retriever(faiss_index_path: str) -> DPRRetriever:
        return DPRRetriever(faiss_index_path)

    @st.cache_resource
    def load_vector_db(embedding_file: str, csv_file: str) -> VectorDatabase:
        return VectorDatabase(embedding_file, csv_file)

    @st.cache_resource
    def load_bypass_llm(model_name: str) -> LLM:
        return LLM(model_name)

    @st.cache_resource
    def load_rag_llm(model_name: str) -> LLM:
        return LLM(model_name)

    @st.cache_resource
    def load_tokenizer() -> AutoTokenizer:
        return AutoTokenizer.from_pretrained(
            "facebook/dpr-question_encoder-single-nq-base"
        )

    # Initialize components
    retriever = load_retriever(faiss_index_path)
    vector_db = load_vector_db(embedding_file, csv_file)
    bypass_llm = load_bypass_llm(bypass_llm_model)
    rag_llm = load_rag_llm(rag_llm_model)
    tokenizer = load_tokenizer()

    st.markdown("---")

    # Data Inspection Section
    st.header("🔍 Data Inspection")

    # Create two columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Slider to select number of rows to display
        num_rows = st.slider(
            "Select number of rows to display",
            min_value=5,
            max_value=min(100, len(vector_db.crime_data)),
            value=10,
            step=5,
        )

        st.subheader(f"Top {num_rows} Crime Data Rows")
        st.dataframe(vector_db.crime_data.head(num_rows))

    with col2:
        # Display crime count statistics
        st.subheader("Crime Count Statistics")
        st.write(vector_db.crime_data["crime_count"].describe())

    # Display the clustered map beneath the crime statistics
    st.subheader("Crime Clusters Map")
    map_data = vector_db.crime_data[["lat", "lon"]].dropna()

    if not map_data.empty:
        # Perform KMeans clustering with 5 clusters
        kmeans = KMeans(n_clusters=5, random_state=42)
        map_data["cluster"] = kmeans.fit_predict(map_data[["lon", "lat"]])

        # Define cluster colors as RGB arrays
        cluster_colors = {
            0: [255, 0, 0],    # Red
            1: [0, 255, 0],    # Green
            2: [0, 0, 255],    # Blue
            3: [255, 165, 0],  # Orange
            4: [128, 0, 128],  # Purple
        }

        # Map cluster labels to RGB colors
        map_data["color"] = map_data["cluster"].map(cluster_colors)

        # Use ScatterplotLayer to visualize clusters with RGB colors
        scatterplot_layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[lon, lat]",
            get_color="color",
            get_radius=100,
            pickable=True,
        )

        # Set the viewport location
        view_state = pdk.ViewState(
            latitude=map_data["lat"].mean(),
            longitude=map_data["lon"].mean(),
            zoom=10,
            pitch=50,
        )

        # Render the deck.gl map
        r = pdk.Deck(layers=[scatterplot_layer], initial_view_state=view_state)
        st.pydeck_chart(r)
    else:
        st.warning("No map data available to display.")

    # ---------------------------
    # Crime Distribution Bar Chart
    # ---------------------------

    st.subheader("Crime Distribution by Crime Type")

    # Get a list of unique years from the data, and add 'Total' option
    years = sorted(vector_db.crime_data['year'].dropna().unique())
    years = [str(int(year)) for year in years]  # Convert years to strings
    years.insert(0, 'Total')  # Add 'Total' option at the beginning
    selected_year = st.selectbox("Select Year or 'Total' for all years", years)

    # Filter data based on selection
    if selected_year == 'Total':
        # Use the entire dataset
        crime_data_filtered = vector_db.crime_data.copy()
    else:
        # Filter data for the selected year
        selected_year_int = int(selected_year)
        crime_data_filtered = vector_db.crime_data[vector_db.crime_data['year'] == selected_year_int]

    if crime_data_filtered.empty:
        st.warning(f"No data available for the selection.")
    else:
        # Count the number of occurrences of each crime type
        crime_type_counts = crime_data_filtered['crime_type'].value_counts().reset_index()
        crime_type_counts.columns = ['crime_type', 'crime_count']

        # Sort the data by 'crime_count' in descending order
        crime_type_counts = crime_type_counts.sort_values(by='crime_count', ascending=False)

        # Limit to top N crime types
        top_n = 10
        crime_type_counts = crime_type_counts.head(top_n)

        # Create a bar chart using Altair
        chart = alt.Chart(crime_type_counts).mark_bar().encode(
            x=alt.X('crime_type:N', sort='-y', axis=alt.Axis(labelAngle=-45)),
            y=alt.Y('crime_count:Q'),
            tooltip=['crime_type', 'crime_count']
        ).properties(
            width='container',
            height=400
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    # User input
    st.header("Enter Your Inquiry")
    user_query = st.text_area("Type your inquiry here:", height=150)

    if st.button("Get Response"):
        if user_query.strip() == "":
            st.warning("⚠️ Please enter a valid inquiry.")
        else:
            with st.spinner("Processing your inquiry..."):
                response = get_response(
                    user_query,
                    retriever,
                    vector_db,
                    bypass_llm,
                    rag_llm,
                    tokenizer,
                    k=num_retrieved_docs,
                )
            if response:
                st.subheader("Response:")
                st.write(response)
            else:
                st.error("Failed to generate a response.")

# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()

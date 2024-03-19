# Sentify - Sentiment Analysis Flask App

Sentify is a simple web application built with Flask that performs sentiment analysis using a Naive Bayes classifier. It allows users to input text and receive predictions about the sentiment of the provided text, whether positive, negative, or neutral.

## Screenshots

Here are some screenshots of the Sentify app in action:

1. **Homepage:**
   ![Homepage](https://github.com/alifhanafiah/sentify/assets/88946724/60232ff6-cf48-4faa-a478-ed93e6bb68a2)


1. **Dataset Page:**
   ![Dataset Page](https://github.com/alifhanafiah/sentify/assets/88946724/f8ea8e65-af75-490e-a986-929177e2d7e5)


1. **Train Model Page:**
   ![Train Model Page](https://github.com/alifhanafiah/sentify/assets/88946724/48da4e81-52f6-467a-ad8e-e30b805a24b1)


1. **Result Page:**
   ![Result Page](https://github.com/alifhanafiah/sentify/assets/88946724/f33c8b5f-d186-44a5-a7d1-29dcf41d9bab)


## Features

- Sentiment analysis using a Naive Bayes classifier.
- User-friendly web interface.
- Quick and easy sentiment predictions.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- Python 3.x installed on your local machine.
- Python packages listed in the `requirements.txt` file.
- Virtualenv installed (you can install it using `pip install virtualenv`).

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/alifhanafiah/sentify.git
   ```

2. Navigate to the project directory:

   ```bash
   cd sentify
   ```

3. Create a virtual environment (replace `venv` with your preferred name):

   ```bash
   virtualenv venv
   ```

4. Activate the virtual environment:

   - On Windows:

   ```bash
   venv\Scripts\activate
   ```

   - On macOS and Linux:

   ```bash
   source venv/bin/activate
   ```

5. Install the required Python packages using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the Flask application:

   ```bash
   python app.py
   ```

2. Open your web browser and navigate to `http://localhost:5000`.

3. Upload a CSV file containing text data for sentiment analysis.

4. After processing, you can see the results as a CSV file, which includes sentiment labels and scores for each entry in the input CSV.

## Configuration

You can customize the behavior of the Sentify app by modifying the `config.py` file.

## License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/alifhanafiah/sentify/blob/main/LICENSE) file for details.

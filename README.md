# MOSDAC HELP BOT

An intelligent chatbot system designed to assist users with MOSDAC (Meteorological and Oceanographic Satellite Data Archival Centre) related queries. This project implements an enhanced RAG (Retrieval Augmented Generation) system for accurate and context-aware responses.

## Features

- **Advanced RAG Implementation**: Utilizes sophisticated retrieval mechanisms for accurate information fetching
- **Context-Aware Responses**: Maintains conversation context for more natural interactions
- **Real-time Processing**: Quick response generation with efficient data processing
- **Web Interface**: Clean and intuitive chat interface for easy interaction
- **Performance Monitoring**: Built-in evaluation metrics for system performance

## Project Structure

```
├── advanced_retriever.py   # Enhanced retrieval mechanisms
├── app.py                 # Main application entry point
├── cache.py              # Caching implementation
├── config.py             # Configuration settings
├── evaluator.py          # Performance evaluation tools
├── processor.py          # Data processing utilities
├── retriever.py          # Base retrieval functionality
├── data/                 # Data directory
│   ├── cache/            # Cache storage
│   ├── evaluation/       # Evaluation results
│   ├── processed/        # Processed data
│   └── raw/              # Raw data files
├── static/               # Static assets
│   ├── css/              # Stylesheets
│   └── images/           # Image assets
└── templates/            # HTML templates
    ├── base.html         # Base template
    ├── chat.html         # Chat interface
    ├── index.html        # Landing page
    └── stats.html        # Statistics display
```

## Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/MOSDAC-HELP-BOT.git
   cd MOSDAC-HELP-BOT
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r req.txt
   ```

4. Set up environment variables:
   Create a `.env` file with necessary configurations

5. Run the application:
   ```bash
   python app.py
   ```

## Usage

1. Access the web interface at `http://localhost:5000`
2. Enter your MOSDAC-related queries in the chat interface
3. Receive accurate, context-aware responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MOSDAC for providing the data and resources
- Contributors and maintainers of the project
- Financial data APIs (list any you're using)
- Firebase/MongoDB (specify your database)

## Target Audience
First-time investors, young professionals, and middle-class households in India seeking trustworthy financial advice.

## Installation
1. Clone this repository
2. Install dependencies: `npm install`
3. Configure environment variables
4. Run the development server: `npm run dev`

## Contribution Guidelines
We welcome contributions! Please fork the repository and create a pull request with your changes.
>>>>>>> 3908da484c91cbeb2eb4ff6e9cad221f4d4a93ac

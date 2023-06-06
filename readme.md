# Price Watcher

Price Watcher is a FastAPI application that scrapes product data from StockX.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

- Python 3.8 or higher
- Docker (optional)

### Installing

1. Clone the repository:

    ```bash
    git clone https://github.com/Louvivien/ProductWatcher.git
    cd price-watcher
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the application:

    ```bash
    uvicorn app:app --reload
    ```

    The application will be available at `http://localhost:8000`.

### Running with Docker

1. Build the Docker image:

    ```bash
    docker build -t price-watcher .
    ```

2. Run the Docker container:

    ```bash
    docker run -p 80:80 price-watcher
    ```

    The application will be available at `http://localhost`.

## Usage

- Visit `http://localhost:8000` (or `http://localhost` if running with Docker) to see the list of products.
- Click on a product to see detailed statistics.

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct, and the process for submitting pull requests to us.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.



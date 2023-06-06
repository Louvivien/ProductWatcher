# Price Watcher

Price Watcher is a web application that displays product data in a table format. It uses Flask for server-side operations and jQuery DataTables for client-side table rendering.

## Setup

1. Install Python 3.6 or later.

2. Install the required Python packages:

    ```bash
    pip install flask
    ```

3. Clone this repository:

    ```bash
    git clone https://github.com/Louvivien/ProductWatcher.git
    cd pricewatcher
    ```

4. Run the Flask server:

    ```bash
    python app.py
    ```

5. Open your web browser and navigate to `http://localhost:5000`.

## Usage

The application displays a table of product data. Each row in the table represents a product, with columns for image, title, color, retail price, average deadstock price, sold, highest bid, lowest ask, and a link to view the product on StockX.

The data is loaded from a JSON file when the Flask server starts. To update the data, replace the JSON file and restart the server.

## Troubleshooting

If the table is not displaying any data, check the following:

- Make sure the JSON file is correctly formatted and contains the expected data.
- Make sure the Flask server is running and accessible.
- Check the JavaScript console in your web browser for any error messages.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)


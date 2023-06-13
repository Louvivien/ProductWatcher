[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]

<p align="center">
  <h3 align="center">Price Watcher</h3>

  <p align="center">
    A web application that displays product data in a table format.
    <br />
    <a href="https://github.com/Louvivien/ProductWatcher"><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="https://productwatcher.fly.dev/">View Demo</a>
    ·
    <a href="https://github.com/Louvivien/ProductWatcher/issues">Report Bug</a>
    ·
    <a href="https://github.com/Louvivien/ProductWatcher/issues">Request Feature</a>
  </p>
</p>

![Python Version][python-image]
![Flask Version][flask-image]
![License][license-image]

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

## Ongoing

Run the scraper when adding a new product to watch

Make several components from app.py



## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

Distributed under the MIT License. See `LICENSE` for more information.

[python-image]: https://img.shields.io/badge/python-v3.6+-blue.svg
[flask-image]: https://img.shields.io/badge/flask-v1.0.2-blue.svg
[license-image]: https://img.shields.io/badge/license-MIT-blue.svg


[contributors-shield]: https://img.shields.io/github/contributors/Louvivien/ProductWatcher.svg?style=for-the-badge
[contributors-url]: https://github.com/Louvivien/ProductWatcher/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Louvivien/ProductWatcher.svg?style=for-the-badge
[forks-url]: https://github.com/Louvivien/ProductWatcher/network/members
[stars-shield]: https://img.shields.io/github/stars/Louvivien/ProductWatcher.svg?style=for-the-badge
[stars-url]: https://github.com/Louvivien/ProductWatcher/stargazers
[issues-shield]: https://img.shields.io/github/issues/Louvivien/ProductWatcher.svg?style=for-the-badge
[issues-url]: https://github.com/Louvivien/ProductWatcher/issues
[license-shield]: https://img.shields.io/github/license/Louvivien/ProductWatcher.svg?style=for-the-badge
[license-url]: https://github.com/Louvivien/ProductWatcher/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/vivienrichaud/
[Next.js]: https://img.shields.io/badge/next.js-000000?style=for-the-badge&logo=nextdotjs&logoColor=white
[Next-url]: https://nextjs.org/
[React.js]: https://img.shields.io/badge/React-20232A?style=for-the-badge&logo=react&logoColor=61DAFB
[React-url]: https://reactjs.org/
[Vue.js]: https://img.shields.io/badge/Vue.js-35495E?style=for-the-badge&logo=vuedotjs&logoColor=4FC08D
[Vue-url]: https://vuejs.org/
[Angular.io]: https://img.shields.io/badge/Angular-DD0031?style=for-the-badge&logo=angular&logoColor=white
[Angular-url]: https://angular.io/
[Svelte.dev]: https://img.shields.io/badge/Svelte-4A4A55?style=for-the-badge&logo=svelte&logoColor=FF3E00
[Svelte-url]: https://svelte.dev/
[Laravel.com]: https://img.shields.io/badge/Laravel-FF2D20?style=for-the-badge&logo=laravel&logoColor=white
[Laravel-url]: https://laravel.com
[Bootstrap.com]: https://img.shields.io/badge/Bootstrap-563D7C?style=for-the-badge&logo=bootstrap&logoColor=white
[Bootstrap-url]: https://getbootstrap.com
[JQuery.com]: https://img.shields.io/badge/jQuery-0769AD?style=for-the-badge&logo=jquery&logoColor=white
[JQuery-url]: https://jquery.com 

<!-- [![MIT License][license-shield]][license-url] -->

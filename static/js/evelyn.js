import { ChartsEmbedSDK } from '@mongodb-js/charts-embed-dom';

async function renderChart() {
  const sdk = new ChartsEmbedSDK({
    baseUrl: 'https://charts.mongodb.com/charts-project-0-kuacs', // Replace with your base URL
  });

  const chart = sdk.createChart({
    chartId: '2cff2ba6-7c65-4a05-af2a-771ea0cdf4e5', // Replace with your chart ID
    // other options
  });

  await chart.render(document.getElementById('chart'));
}

renderChart();

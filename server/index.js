const express = require('express');
const app = express();
const PORT = process.env.PORT || 5000;

app.get('/', (req, res) => {
  res.send('Express backend is running.');
});

// Placeholder endpoints for neural network and storage
app.get('/api/network', (req, res) => {
  res.json({ message: 'Neural network data endpoint.' });
});

app.get('/api/storage', (req, res) => {
  res.json({ message: 'Storage/data endpoint.' });
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});

import React from 'react';
import { BrowserRouter as Router, Route, Routes, Link } from 'react-router-dom';

function Home() {
  return <div><h1>Home</h1><p>Welcome! Choose a page:</p><ul><li><Link to="/network">Neural Network Graph</Link></li><li><Link to="/storage">Storage/Data</Link></li></ul></div>;
}

function NetworkGraph() {
  return <div><h1>Neural Network Graph</h1><p>Network visualization will go here.</p></div>;
}

function Storage() {
  return <div><h1>Storage/Data</h1><p>File and data management will go here.</p></div>;
}

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/network" element={<NetworkGraph />} />
        <Route path="/storage" element={<Storage />} />
      </Routes>
    </Router>
  );
}

export default App;

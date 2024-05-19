import React from 'react';
import './styles/App.css';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';
import Connections from './pages/Connections';

const App = () => {
    return (
        <BrowserRouter>
            <Navbar></Navbar>
            <Routes>
                <Route element={<Home />} path="/" />
                <Route element={<Connections />} path="/connections" />
            </Routes>
        </BrowserRouter>
    );
};

export default App;

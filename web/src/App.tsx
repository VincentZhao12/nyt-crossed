import React from 'react';
import './styles/App.css';
import { BrowserRouter, Route, Routes } from 'react-router-dom';
import Navbar from './components/Navbar';
import Home from './pages/Home';

const App = () => {
    return (
        <BrowserRouter>
            <Navbar></Navbar>
            <Routes>
                <Route element={<Home />} path="/" />
            </Routes>
        </BrowserRouter>
    );
};

export default App;
